# -*- coding: utf-8 -*-
"""
Step 1.2  MI / NMI edge weights (single clean pass)
---------------------------------------------------
"""

import os, time, itertools, json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from tqdm import tqdm
from rasterio.features import rasterize
from rasterio.transform import Affine
from pyproj import CRS


# =========================
# Paths (project-relative)
# =========================
HERE = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("SES_DATA_ROOT", HERE / "data")).resolve()
OUT_ROOT  = Path(os.environ.get("SES_OUT_ROOT",  HERE / "outputs")).resolve()

ZARR_IN   = OUT_ROOT / "data_cube" / "ses_cube_1km_albers.zarr"
ADMIN_SHP = DATA_ROOT / "geometry" / "M_IM_Albers.shp"
ZONE_FIELD = "ENG_NAME"   # change if your shapefile uses a different column
SCALE_TAG  = "M_IM"

OUT_FILE = OUT_ROOT / "network" / f"edges_{SCALE_TAG}_MI_NMI.parquet"
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)


# =========================
# Variables (whitelist)
# =========================
ALLOWED_SET = {
    # D
    "PRE","PET","T","AI","SPEI6","WS2",
    # E
    "FVC","NIRv","NDMI","SM","AET",
    # S
    "FS","FD","GS","GD","WEPS",
    # H
    "POP","NTL","LD","CP","GP",
}


# =========================
# Parameters (MI/NMI only)
# =========================
TIME_RANGE = ("2000-01-01", "2023-12-31")
MIN_SAMPLES = 600

# Adaptive quantile binning
Q_MIN, Q_MAX = 6, 12
M_TARGET_PER_BIN = 40


def stamp(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def open_zarr_any(path: Path) -> xr.Dataset:
    try:
        return xr.open_zarr(str(path), consolidated=True)
    except Exception:
        return xr.open_zarr(str(path), consolidated=False)


def get_grid(ds: xr.Dataset):
    if "crs_wkt" not in ds.attrs or "transform" not in ds.attrs:
        raise ValueError("Missing grid metadata in Zarr: ds.attrs['crs_wkt'] or ds.attrs['transform']")

    crs = CRS.from_wkt(ds.attrs["crs_wkt"])
    transform = Affine.from_gdal(*json.loads(ds.attrs["transform"]))

    if "y" not in ds.coords or "x" not in ds.coords:
        raise ValueError("Missing y/x coordinates in Zarr.")

    h, w = int(ds.sizes["y"]), int(ds.sizes["x"])
    y = ds["y"].values
    x = ds["x"].values
    return crs, transform, (h, w), y, x


def rasterize_zones(admin_shp: Path, zone_field: str, crs, transform, shape, y, x):
    gdf = gpd.read_file(str(admin_shp))
    if gdf.crs is None or CRS.from_user_input(gdf.crs).to_wkt() != crs.to_wkt():
        gdf = gdf.to_crs(crs.to_wkt())

    if zone_field not in gdf.columns:
        raise ValueError(f"ZONE_FIELD='{zone_field}' not found in {admin_shp.name}")

    zone_key = gdf[zone_field].astype(str).fillna("NA")
    zone_id, uniques = pd.factorize(zone_key, sort=True)
    gdf["zone_id"] = zone_id.astype(np.int32)
    gdf["zone_key"] = zone_key

    shapes_iter = ((geom, int(zid)) for geom, zid in zip(gdf.geometry, gdf["zone_id"]))
    zone_arr = rasterize(
        shapes=shapes_iter,
        out_shape=shape,
        transform=transform,
        fill=-1,
        dtype="int32",
        all_touched=False
    )

    zone_da = xr.DataArray(zone_arr, dims=("y","x"), coords={"y": y, "x": x}, name="zone_id")
    meta = gdf.drop(columns="geometry").drop_duplicates(subset=["zone_id"]).set_index("zone_id")
    return zone_da, meta


def discretize_quantile_1d(x: np.ndarray, q: int):
    x = np.asarray(x)
    ok = np.isfinite(x)
    xv = x[ok]
    if xv.size == 0:
        return None

    edges = np.unique(np.nanquantile(xv, np.linspace(0, 1, q + 1)))
    if edges.size < 3:
        return None

    xb = np.full(x.shape, -1, dtype=np.int16)
    xb[ok] = np.digitize(xv, edges[1:-1], right=False)
    return xb


def mi_nmi_from_bins(xb: np.ndarray, yb: np.ndarray, q: int):
    mask = (xb >= 0) & (yb >= 0)
    n_eff = int(mask.sum())
    if n_eff < MIN_SAMPLES:
        return np.nan, np.nan, 0

    x = xb[mask].astype(np.int32)
    y = yb[mask].astype(np.int32)

    lin = x * q + y
    H = np.bincount(lin, minlength=q*q).reshape(q, q).astype(float)
    if H.sum() == 0:
        return np.nan, np.nan, 0

    Hx = H.sum(1)
    Hy = H.sum(0)

    def entropy(counts):
        s = counts.sum()
        if s <= 0:
            return 0.0
        p = counts / s
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())

    HX  = entropy(Hx)
    HY  = entropy(Hy)
    HXY = entropy(H.ravel())

    MI  = HX + HY - HXY
    NMI = float(MI / np.sqrt(HX * HY)) if HX > 0 and HY > 0 else np.nan
    return float(MI), float(NMI), n_eff


def main():
    t0 = time.perf_counter()

    if not ZARR_IN.exists():
        raise FileNotFoundError(f"Zarr not found:\n  {ZARR_IN}")
    if not ADMIN_SHP.exists():
        raise FileNotFoundError(f"Admin shapefile not found:\n  {ADMIN_SHP}")

    stamp(f"[0] Reading cube: {ZARR_IN.name}")
    ds = open_zarr_any(ZARR_IN)

    vars_use = [v for v in ds.data_vars if str(v) in ALLOWED_SET]
    if not vars_use:
        raise RuntimeError("No variables matched ALLOWED_SET in the cube.")

    crs, transform, shape, y, x = get_grid(ds)

    stamp(f"[1] Rasterizing zones: {ADMIN_SHP.name}")
    zone_da, meta = rasterize_zones(ADMIN_SHP, ZONE_FIELD, crs, transform, shape, y, x)
    zone = zone_da.values
    zone_ids = np.sort(np.unique(zone[zone >= 0]))
    stamp(f"[zones] {len(zone_ids)} zones")

    # time selection
    times = pd.to_datetime(ds["time"].values)
    t0r, t1r = pd.Timestamp(TIME_RANGE[0]), pd.Timestamp(TIME_RANGE[1])
    times = times[(times >= t0r) & (times <= t1r)]
    if times.size == 0:
        raise RuntimeError("No time slices within TIME_RANGE.")

    n_pairs = len(vars_use) * (len(vars_use) - 1) // 2
    est = int(len(zone_ids) * len(times) * n_pairs)
    stamp(f"[estimate] years={len(times)} vars={len(vars_use)} pairs={n_pairs} edges~{est:,}")

    edges = []
    pbar = tqdm(total=est, desc="MI/NMI", unit="edge")

    for zid in zone_ids:
        yy, xx = np.where(zone == zid)
        if yy.size < MIN_SAMPLES:
            pbar.update(len(times) * n_pairs)
            continue

        # bbox crop
        y0, y1 = yy.min(), yy.max() + 1
        x0, x1 = xx.min(), xx.max() + 1
        zmask = (zone[y0:y1, x0:x1] == zid)

        zkey = meta.loc[int(zid), "zone_key"] if int(zid) in meta.index else str(zid)

        for t in times:
            sub = ds[vars_use].isel(y=slice(y0, y1), x=slice(x0, x1)).sel(time=t).load()

            n_pix = int(zmask.sum())
            q_this = max(Q_MIN, min(Q_MAX, int(np.sqrt(max(n_pix, 1) / M_TARGET_PER_BIN))))

            bins = {}
            for v in vars_use:
                x1d = sub[v].values[zmask].astype(float)
                bins[v] = discretize_quantile_1d(x1d, q=q_this)

            for i, j in itertools.combinations(vars_use, 2):
                xb = bins.get(i)
                yb = bins.get(j)
                if xb is None or yb is None:
                    pbar.update(1)
                    continue

                mi, nmi, n = mi_nmi_from_bins(xb, yb, q=q_this)
                if not np.isfinite(nmi):
                    pbar.update(1)
                    continue

                edges.append({
                    "scale_tag": SCALE_TAG,
                    "zone_id": int(zid),
                    "zone_key": str(zkey),
                    "time": pd.Timestamp(t),
                    "var_i": str(i),
                    "var_j": str(j),
                    "n": int(n),
                    "mi": float(mi),
                    "nmi": float(nmi),
                    "q_used": int(q_this),
                })
                pbar.update(1)

    pbar.close()

    df = pd.DataFrame(edges)
    if df.empty:
        raise RuntimeError("No MI/NMI edges were computed. Check MIN_SAMPLES, TIME_RANGE, and data coverage.")

    df.to_parquet(OUT_FILE)
    stamp(f"[done] wrote: {OUT_FILE}")
    stamp(f"[done] elapsed: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
