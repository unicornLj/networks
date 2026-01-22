# -*- coding: utf-8 -*-
"""
Step 1.1  Build a 1 km Albers-projected data cube
------------------------------------------------
"""

import os, sys, re, glob, json, time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.crs import CRS
from tqdm import tqdm
from numcodecs import Blosc


# =========================================================
# Paths
# =========================================================
HERE = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("SES_DATA_ROOT", HERE / "data")).resolve()
OUT_ROOT  = (HERE / "outputs").resolve()

TEMPLATE_RASTER = DATA_ROOT / "1.1 PRE_sum_2000_2024" / "pre_sum_mm_2000.tif"

ZARR_OUT = OUT_ROOT / "data_cube" / "ses_cube_1km_albers.zarr"
QA_OUT   = OUT_ROOT / "data_cube" / "qa_summary.csv"

CHUNKS = {"time": 1, "y": 1024, "x": 1024}
COMPRESSOR = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)


# =========================================================
# Variable sources (names consistent with later steps)
# =========================================================
VARS: Dict[str, str] = {
    # C
    "PRE":   str(DATA_ROOT / "1.1 PRE_sum_2000_2023"   / "pre_sum_mm_*.tif"),
    "PET":   str(DATA_ROOT / "1.2 PET_sum_2000_2023"   / "PET_sum_mm_*.tif"),
    "AI":    str(DATA_ROOT / "1.3 AI_2000_2023"        / "AI_*.tif"),
    "T":     str(DATA_ROOT / "1.4 T_mean_2000_2023"    / "T_mean_C_*.tif"),
    "SPEI6": str(DATA_ROOT / "1.5 SPEI_2000_2023"      / "SPEI_*.tif"),
    "WS2":   str(DATA_ROOT / "1.6 WS2_mean_2000_2023"  / "WS2_mean_*.tif"),

    # E
    "FVC":   str(DATA_ROOT / "2.1 FVC_2000_2023"       / "FVC_*.tif"),
    "NIRv":  str(DATA_ROOT / "2.2 NIRv_2000_2023"      / "NIRv_*.tif"),
    "NDMI":  str(DATA_ROOT / "2.3 NDMI_2000_2023"      / "NDMI_*.tif"),
    "SM":    str(DATA_ROOT / "2.4 SM_2000_2023"        / "SM_root_*.tif"),
    "AET":   str(DATA_ROOT / "2.5 AET_2000_2023"       / "AET_*.tif"),

    # A
    "POP":   str(DATA_ROOT / "3.1 POP_2000_2023"       / "POP_1km__*.tif"),
    "NTL":   str(DATA_ROOT / "3.2 NTL_1993_2023"       / "NTL_1km__*.tif"),
    "LD":    str(DATA_ROOT / "3.3 SU2_2000_2023"       / "SU2_1km_*.tif"),
    "CP":    str(DATA_ROOT / "3.4 Farmland_2000_2023"  / "Farmland_*.tif"),
    "GP":    str(DATA_ROOT / "3.5 Grassland_2000_2023" / "Grassland_*.tif"),

    # ES
    "GS":    str(DATA_ROOT / "4.1 GS_2000_2023"        / "GS_kg_ha_*.tif"),
    "GD":    str(DATA_ROOT / "4.2 GD_2000_2023"        / "GD_kg_ha_*.tif"),
    "FS":    str(DATA_ROOT / "4.3 FS_2000_2023"        / "FS_1km_*.tif"),
    "FD":    str(DATA_ROOT / "4.4 FD_2000_2023"        / "FD_kg_ha_*.tif"),
    "WECS":  str(DATA_ROOT / "4.5 WECS_2000_2023"      / "WECS_t_ha_*.tif"),
}

# Only keep the whitelist used in the current workflow
INCLUDE_ONLY = {
    "PRE","PET","AI","T","SPEI6","WS2",
    "FVC","NIRv","NDMI","SM","AET",
    "POP","NTL","LD","GP","CP",
    "GS","GD","FS","FD","WECS"
}

# Resampling: continuous/proportion -> bilinear (change specific vars to "nearest" if needed)
RESAMPLE_KIND = {v: "bilinear" for v in INCLUDE_ONLY}

YEAR_REGEX = re.compile(r"(19|20)\d{2}")


def stamp(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def get_template(path: Path) -> Tuple[CRS, Affine, Tuple[int, int], xr.DataArray]:
    da = rxr.open_rasterio(str(path), masked=True).squeeze("band", drop=True)
    if not da.rio.crs:
        raise ValueError(f"Template raster has no CRS: {path}")
    return da.rio.crs, da.rio.transform(), (da.sizes["y"], da.sizes["x"]), da


def parse_year_from_name(fp: str) -> Optional[int]:
    m = YEAR_REGEX.search(os.path.basename(fp))
    return int(m.group(0)) if m else None


def _resample(var: str) -> Resampling:
    return Resampling.nearest if RESAMPLE_KIND.get(var, "bilinear") == "nearest" else Resampling.bilinear


def open_and_align(
    fp: str,
    template_crs: CRS,
    template_transform: Affine,
    template_shape: Tuple[int, int],
    template_da: xr.DataArray,
    var: str
) -> xr.DataArray:
    da = rxr.open_rasterio(fp, masked=True).squeeze("band", drop=True)

    # nodata -> NaN
    if "nodata" in da.attrs and da.attrs["nodata"] is not None:
        da = da.where(da != da.attrs["nodata"])
    else:
        da = da.where(np.isfinite(da))

    resamp = _resample(var)

    # Align to template
    if da.rio.crs != template_crs:
        da = da.rio.reproject(
            template_crs,
            transform=template_transform,
            shape=template_shape,
            resampling=resamp
        )
    else:
        da = da.rio.reproject_match(template_da, resampling=resamp)

    return da.astype("float32").where(np.isfinite(da), other=np.nan)


def stack_variable(
    var_name: str,
    pattern: str,
    template_crs: CRS,
    template_transform: Affine,
    template_shape: Tuple[int, int],
    template_da: xr.DataArray
) -> xr.DataArray:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"[{var_name}] No files found: {pattern}")

    t0 = time.perf_counter()
    records, skipped = [], []

    for fp in tqdm(files, desc=f"[{var_name}] read/align", leave=False):
        yr = parse_year_from_name(fp)
        if yr is None:
            skipped.append(fp)
            continue

        da = open_and_align(fp, template_crs, template_transform, template_shape, template_da, var_name)
        da = da.rename({"y": "y", "x": "x"}).expand_dims(time=[pd.Timestamp(f"{yr}-12-31")])
        da.name = var_name
        records.append(da)

    if skipped:
        stamp(f"[note] {var_name}: skipped {len(skipped)} files (year not parsed).")

    if not records:
        raise RuntimeError(f"[{var_name}] No slices were stacked successfully.")

    out = xr.concat(records, dim="time").sortby("time")
    out.attrs.update({"long_name": var_name, "units": "unknown"})
    stamp(f"[done] {var_name} stacked: {tuple(out.shape)} | {time.perf_counter()-t0:.1f}s")
    return out


def main():
    # Basic checks
    if not TEMPLATE_RASTER.exists():
        raise FileNotFoundError(
            f"Template raster not found:\n  {TEMPLATE_RASTER}\n"
            f"DATA_ROOT is:\n  {DATA_ROOT}\n"
            f"Make sure the reviewer placed the data under ./data with the expected folder structure."
        )

    ZARR_OUT.parent.mkdir(parents=True, exist_ok=True)
    QA_OUT.parent.mkdir(parents=True, exist_ok=True)

    t_run0 = time.perf_counter()

    manifest = {
        "vars": {},
        "cube": {},
        "qa_path": None,
        "run": {"start": datetime.now().isoformat()},
        "paths": {
            "DATA_ROOT": str(DATA_ROOT),
            "OUT_ROOT": str(OUT_ROOT),
            "TEMPLATE_RASTER": str(TEMPLATE_RASTER),
            "ZARR_OUT": str(ZARR_OUT),
            "QA_OUT": str(QA_OUT),
        }
    }

    # 1) Template
    stamp("[1/4] Reading template...")
    crs, transform, shape, tmpl_da = get_template(TEMPLATE_RASTER)
    stamp(f"[template] CRS={crs.to_string()}, shape={shape}")

    # 2) Stack variables (whitelist only)
    stamp("[2/4] Stacking variables...")
    data_vars = {}
    all_times = set()

    for var, pattern in tqdm(VARS.items(), desc="variables"):
        if var not in INCLUDE_ONLY:
            continue
        try:
            da = stack_variable(var, pattern, crs, transform, shape, tmpl_da).chunk(CHUNKS)
            data_vars[var] = da
            all_times.update(pd.to_datetime(da["time"].values).tolist())
            manifest["vars"][var] = {
                "n_slices": int(da.sizes["time"]),
                "time_min": str(pd.to_datetime(da["time"].values).min()),
                "time_max": str(pd.to_datetime(da["time"].values).max()),
                "shape": [int(da.sizes["time"]), int(da.sizes["y"]), int(da.sizes["x"])],
                "resampling": RESAMPLE_KIND.get(var, "bilinear")
            }
        except Exception as e:
            stamp(f"[skip] {var}: {e}")

    if not data_vars:
        raise RuntimeError("No variables stacked successfully. Check ./data structure and file name patterns.")

    # 3) Build Dataset (align time axis)
    time_index = pd.to_datetime(sorted(set(all_times)))
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": time_index,
            "y": list(data_vars.values())[0]["y"].values,
            "x": list(data_vars.values())[0]["x"].values,
        }
    )

    # QA summary (NaN ratio / min / max)
    qa_rows = []
    for v, da in data_vars.items():
        times = pd.to_datetime(da["time"].values)
        for t in times:
            arr = da.sel(time=t).values
            qa_rows.append({
                "var": v,
                "time": str(pd.to_datetime(t).date()),
                "nan_ratio": float(np.isnan(arr).mean()),
                "min": float(np.nanmin(arr)),
                "max": float(np.nanmax(arr))
            })

    qa = pd.DataFrame(qa_rows)
    qa.to_csv(str(QA_OUT), index=False, encoding="utf-8-sig")
    manifest["qa_path"] = str(QA_OUT)

    # Metadata
    ds.attrs.update({
        "title": "SES Data Cube (1km Albers)",
        "summary": "Social–Ecological variables stacked as a spatiotemporal data cube",
        "Conventions": "CF-1.8",
        "grid_mapping": "spatial_ref",
        "crs_wkt": crs.to_wkt(),
        "transform": json.dumps(transform.to_gdal()),
        "created": datetime.now().isoformat()
    })

    # 4) Write Zarr
    stamp("[3/4] Writing Zarr...")
    encoding = {v: {"compressor": COMPRESSOR, "dtype": "float32"} for v in ds.data_vars}
    ds.astype("float32").to_zarr(str(ZARR_OUT), mode="w", encoding=encoding, consolidated=True)
    stamp(f"[write] Zarr done: {ZARR_OUT}")

    # 5) Manifest
    manifest["cube"] = {
        "zarr_path": str(ZARR_OUT),
        "dims": {k: int(v) for k, v in ds.dims.items()},
        "data_vars": sorted(list(ds.data_vars)),
        "coords": sorted(list(ds.coords))
    }
    manifest["run"]["end"] = datetime.now().isoformat()
    manifest["run"]["elapsed_sec"] = round(time.perf_counter() - t_run0, 1)

    with open(str(ZARR_OUT.parent / "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    stamp("[4/4] Done. Zarr + QA written.")


if __name__ == "__main__":
    main()
