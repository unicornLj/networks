# -*- coding: utf-8 -*-
import os, re, math, warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx


# =========================
# Basic config
# =========================
HERE = Path(__file__).resolve().parent
OUT_ROOT = Path(os.environ.get("SES_OUT_ROOT", HERE / "outputs")).resolve()

SCALE_TAG = "M_IM"  # change if needed
EDGES_FILE = OUT_ROOT / "network" / f"edges_{SCALE_TAG}_MI_NMI.parquet"

TS_DIR = OUT_ROOT / "network" / "topology_timeseries"
TS_DIR.mkdir(parents=True, exist_ok=True)

TOPK_PER_YEAR = 60
YEAR_RANGE = (2000, 2023)
MIN_EDGES_FOR_Q = 1
EPS = 1e-9


# =========================
# Layer mapping (C/E/ES/A)
# =========================
LAYER_MAP = {
    # Climate
    "PRE":"C","PET":"C","T":"C","AI":"C","SPEI6":"C","WS2":"C",
    # Ecosystem
    "FVC":"E","NIRv":"E","NDMI":"E","SM":"E","AET":"E",
    # Ecosystem services
    "FS":"ES","FD":"ES","GS":"ES","GD":"ES","WECS":"ES",
    # Human activities
    "POP":"A","NTL":"A","LD":"A","CP":"A","GP":"A",
}


def stamp(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def base_var(name: str) -> str:
    # keep your naming rule: strip _anom/_detr if any
    return re.sub(r'_(anom|detr)$', '', str(name), flags=re.IGNORECASE)


def layer_of(node: str) -> str:
    return LAYER_MAP.get(base_var(node), "U")


def keep_years(df: pd.DataFrame, yr_range):
    if yr_range is None:
        return df
    y0, y1 = int(yr_range[0]), int(yr_range[1])
    yy = pd.to_datetime(df["time"]).dt.year
    return df[(yy >= y0) & (yy <= y1)].copy()


def filter_topk_by_group(df: pd.DataFrame, topk: int):
    if topk is None or int(topk) <= 0:
        return df
    df = df.sort_values("nmi", ascending=False)
    return df.head(int(topk))


def build_graph(df_edges: pd.DataFrame) -> nx.Graph:
    """
    df_edges must contain: var_i, var_j, nmi
    """
    G = nx.Graph()
    for _, r in df_edges.iterrows():
        u = str(r["var_i"])
        v = str(r["var_j"])
        if u == v:
            continue
        w = float(r["nmi"]) if pd.notna(r["nmi"]) else 0.0
        if w <= 0:
            continue
        if G.has_edge(u, v):
            if w > float(G[u][v].get("weight", 0.0)):
                G[u][v]["weight"] = w
        else:
            G.add_edge(u, v, weight=w)

    for n in G.nodes():
        G.nodes[n]["layer"] = layer_of(n)

    return G


def compute_eglob(G: nx.Graph) -> float:
    N = G.number_of_nodes()
    M = G.number_of_edges()
    if N <= 1 or M <= 0:
        return np.nan

    Gb = nx.Graph()
    Gb.add_nodes_from(G.nodes())
    for u, v, d in G.edges(data=True):
        w = float(d.get("weight", 0.0))
        if w > 0:
            Gb.add_edge(u, v, length=1.0 / (w + EPS))

    lengths = dict(nx.all_pairs_dijkstra_path_length(Gb, weight="length"))

    s_sum = 0.0
    nodes = list(G.nodes())
    for u in nodes:
        dist_u = lengths.get(u, {})
        for v in nodes:
            if u == v:
                continue
            d = dist_u.get(v, np.inf)
            if np.isfinite(d) and d > 0:
                s_sum += 1.0 / d

    return float(s_sum / (N * (N - 1))) if N > 1 else np.nan


def compute_syci(G: nx.Graph):
    M = G.number_of_edges()
    if M <= 0:
        return np.nan, np.nan

    intra = 0
    es_es = 0
    for u, v in G.edges():
        lu = G.nodes[u].get("layer", "U")
        lv = G.nodes[v].get("layer", "U")
        if lu == lv:
            intra += 1
            if lu == "ES":
                es_es += 1

    cross = M - intra
    syci = 1.0 - (intra / max(M, 1))

    denom = M - es_es
    syci_noss = (cross / denom) if denom > 0 else np.nan
    return float(syci), float(syci_noss)


def compute_intra_inter_strength(G: nx.Graph):
    intra_s, inter_s, inter_share = {}, {}, {}
    for n in G.nodes():
        ln = G.nodes[n].get("layer", "U")
        s_intra = 0.0
        s_inter = 0.0
        for nb, d in G[n].items():
            w = float(d.get("weight", 0.0))
            if G.nodes[nb].get("layer", "U") == ln:
                s_intra += w
            else:
                s_inter += w
        intra_s[n] = float(s_intra)
        inter_s[n] = float(s_inter)
        tot = s_intra + s_inter
        inter_share[n] = float(s_inter / tot) if tot > 0 else np.nan
    return intra_s, inter_s, inter_share


def dicts_to_wide(ts_dict: dict, index_name="node"):
    years = sorted(ts_dict.keys())
    nodes = sorted({n for y in years for n in ts_dict[y].keys()})
    out = pd.DataFrame(index=nodes, columns=years, dtype=float)
    out.index.name = index_name
    for y in years:
        for n, v in ts_dict[y].items():
            out.at[n, y] = v
    return out


def main():
    if not EDGES_FILE.exists():
        raise SystemExit(f"Edges file not found:\n  {EDGES_FILE}")

    df = pd.read_parquet(EDGES_FILE)
    if df.empty:
        raise SystemExit("Edges table is empty.")

    need_cols = {"zone_key", "time", "var_i", "var_j", "nmi"}
    miss = need_cols - set(df.columns)
    if miss:
        raise SystemExit(f"Missing columns in edges table: {sorted(miss)}")

    df["time"] = pd.to_datetime(df["time"])
    df = keep_years(df, YEAR_RANGE)

    # group by zone-year
    df["year"] = df["time"].dt.year.astype(int)
    groups = df.groupby(["zone_key", "year"], sort=True)

    # collectors
    global_rows = []
    strength_ts = {}       # zone -> {year -> {node -> value}}
    betw_ts = {}
    eig_ts = {}
    intra_ts = {}
    inter_ts = {}
    inter_share_ts = {}

    stamp(f"groups={groups.ngroups} | topK={TOPK_PER_YEAR}")

    for (zk, yr), g0 in groups:
        g0 = g0.dropna(subset=["nmi"]).copy()
        if g0.empty:
            continue

        g = filter_topk_by_group(g0, TOPK_PER_YEAR)
        G = build_graph(g)

        N = G.number_of_nodes()
        M = G.number_of_edges()

        density = nx.density(G) if N > 1 else 0.0
        kbar = (2.0 * M / N) if N > 0 else 0.0

        # modularity Q
        if M >= MIN_EDGES_FOR_Q:
            try:
                comms = list(nx.algorithms.community.greedy_modularity_communities(G, weight="weight"))
                Q = float(nx.algorithms.community.modularity(G, comms, weight="weight"))
            except Exception:
                Q = np.nan
        else:
            Q = np.nan

        Eglob = compute_eglob(G)
        SyCI, SyCI_noss = compute_syci(G)

        # node metrics
        strength = {n: sum(d.get("weight", 0.0) for _, _, d in G.edges(n, data=True)) for n in G.nodes()}
        intra_s, inter_s, inter_share = compute_intra_inter_strength(G)

        if M > 0:
            Gb = nx.Graph()
            Gb.add_nodes_from(G.nodes())
            for u, v, d in G.edges(data=True):
                w = float(d.get("weight", 0.0))
                if w > 0:
                    Gb.add_edge(u, v, length=1.0 / (w + EPS))
            btw = nx.betweenness_centrality(Gb, weight="length", normalized=True)
        else:
            btw = {n: 0.0 for n in G.nodes()}

        try:
            eig = nx.eigenvector_centrality_numpy(G, weight="weight") if M > 0 else {n: 0.0 for n in G.nodes()}
        except Exception:
            eig = {n: np.nan for n in G.nodes()}

        # store node timeseries
        strength_ts.setdefault(zk, {})[yr] = strength
        betw_ts.setdefault(zk, {})[yr] = btw
        eig_ts.setdefault(zk, {})[yr] = eig
        intra_ts.setdefault(zk, {})[yr] = intra_s
        inter_ts.setdefault(zk, {})[yr] = inter_s
        inter_share_ts.setdefault(zk, {})[yr] = inter_share

        # store global timeseries row
        global_rows.append({
            "scale_tag": SCALE_TAG,
            "zone_key": str(zk),
            "year": int(yr),
            "density": float(density),
            "kbar": float(kbar),
            "Q": float(Q) if np.isfinite(Q) else np.nan,
            "Eglob": float(Eglob) if np.isfinite(Eglob) else np.nan,
            "SyCI": float(SyCI) if np.isfinite(SyCI) else np.nan,
            "SyCI_noss": float(SyCI_noss) if np.isfinite(SyCI_noss) else np.nan,
            "n_nodes": int(N),
            "n_edges": int(M),
        })

    if not global_rows:
        raise SystemExit("No zone-year graphs were built. Check YEAR_RANGE / TOPK_PER_YEAR / edges file.")

    global_df = pd.DataFrame(global_rows).sort_values(["zone_key", "year"])
    long_path = TS_DIR / f"topology_timeseries_long_{SCALE_TAG}.csv"
    global_df.to_csv(long_path, index=False, encoding="utf-8-sig")

    metrics = ["density", "kbar", "Q", "Eglob", "SyCI", "SyCI_noss", "n_nodes", "n_edges"]
    wide_blocks = []
    for m in metrics:
        tmp = global_df.pivot(index="zone_key", columns="year", values=m)
        tmp.columns = [f"{m}_{c}" for c in tmp.columns]
        wide_blocks.append(tmp)
    wide_df = pd.concat(wide_blocks, axis=1).reset_index()
    wide_path = TS_DIR / f"topology_timeseries_wide_{SCALE_TAG}.csv"
    wide_df.to_csv(wide_path, index=False, encoding="utf-8-sig")


    def export_node_ts(ts_by_zone: dict, fname: str):
        rows = []
        for zk, yd in ts_by_zone.items():
            wide = dicts_to_wide(yd, index_name="node").reset_index()
            wide.insert(0, "zone_key", str(zk))
            rows.append(wide)
        out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        out.to_csv(TS_DIR / fname, index=False, encoding="utf-8-sig")

    export_node_ts(strength_ts,    f"strength_nodes_{SCALE_TAG}.csv")
    export_node_ts(betw_ts,        f"betweenness_nodes_{SCALE_TAG}.csv")
    export_node_ts(eig_ts,         f"eigenvector_nodes_{SCALE_TAG}.csv")
    export_node_ts(intra_ts,       f"intra_strength_nodes_{SCALE_TAG}.csv")
    export_node_ts(inter_ts,       f"inter_strength_nodes_{SCALE_TAG}.csv")
    export_node_ts(inter_share_ts, f"inter_share_nodes_{SCALE_TAG}.csv")

    stamp(f"saved: {long_path}")
    stamp(f"saved: {wide_path}")
    stamp(f"saved node tables under: {TS_DIR}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
