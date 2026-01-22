# -*- coding: utf-8 -*-
import os
import math
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import patheffects as pe
from datetime import datetime
from pathlib import Path


# ----------------- Layering and colors (C/E/ES/A) -----------------
LAYER_MAP = {
    "PRE":"C","PET":"C","T":"C","AI":"C","SPEI6":"C","WS2":"C",
    "FVC":"E","NIRv":"E","NDMI":"E","SM":"E","AET":"E",
    "FS":"ES","FD":"ES","GS":"ES","GD":"ES","WECS":"ES",
    "POP":"A","NTL":"A","LD":"A","CP":"A","GP":"A",
}
COLOR_MAP = {"C":"#3C9BC9","E":"#B0D6A9","ES":"#FAA26F","A":"#7B6C9B","U":"#7f7f7f"}

INTRA_GRAY   = "#BFBFBF"
INTER_YELLOW = "#F2C84C"

NODE_BASE, NODE_SCALE = 50, 500
LABEL_OFFSET, LABEL_SIZE = 0.02, 8

DISPLAY_CODE = {"D": "C", "E": "E", "S": "ES", "H": "A", "U": "U"}
DISPLAY_NAME = {
    "C": "Climatic",
    "E": "Ecosystem",
    "ES": "Services",
    "A": "Human activities",
    "U": "Unmapped",
}
LEGEND_ORDER = ["C", "E", "ES", "A", "U"]
OUT_ROOT = Path(os.getenv("SEN2DASH_OUTROOT", "../../output/sen2dash"))

def stamp(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def base_var(name: str) -> str:
    # keep your naming rule: strip _anom/_detr if exists
    import re
    return re.sub(r'_(anom|detr)$', '', str(name), flags=re.IGNORECASE)


def circle_positions(names, center=(0.0, 0.0), r=0.25, rotation_deg=0.0):
    cx, cy = center
    rot = math.radians(rotation_deg)
    m = max(len(names), 1)
    pos = {}
    for i, n in enumerate(names):
        theta = rot + 2 * math.pi * i / m
        pos[n] = (cx + r * math.cos(theta), cy + r * math.sin(theta))
    return pos


def compute_strength(G: nx.Graph):
    return {n: sum(d.get("weight", 0.0) for _, _, d in G.edges(n, data=True)) for n in G.nodes()}


def build_graph_from_edges(df: pd.DataFrame) -> nx.Graph:
    """
    df must contain: var_i, var_j, nmi
    optional: mi, n, q_used
    """
    need = {"var_i", "var_j", "nmi"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Missing columns: {sorted(miss)}")

    G = nx.Graph()
    for _, r in df.iterrows():
        u = str(r["var_i"])
        v = str(r["var_j"])
        w = float(r["nmi"]) if pd.notna(r["nmi"]) else 0.0
        if u == v:
            continue
        if G.has_edge(u, v):
            # keep the stronger one if duplicates exist
            if w > float(G[u][v].get("weight", 0.0)):
                G[u][v]["weight"] = w
        else:
            G.add_edge(u, v, weight=w)

    for n in G.nodes():
        lyr = LAYER_MAP.get(base_var(n), "U")
        G.nodes[n]["layer"] = lyr
        G.nodes[n]["color"] = COLOR_MAP.get(lyr, "#7f7f7f")

    return G


def draw_graph(G, title_main, png_path, note_text=None, fig_size=(5, 5), dpi=1200):
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
    ax.axis("off")

    init_pos, fixed_nodes = {}, []

    # 1) D/E initial circle (left)
    DE_nodes = [n for n in G.nodes() if G.nodes[n].get("layer") in ("D", "E")]
    if DE_nodes:
        init_pos.update(circle_positions(DE_nodes, center=(-0.65, 0.05), r=0.45, rotation_deg=90))

    def _center_radius(pts: dict, nodes: list):
        xs = [pts[n][0] for n in nodes if n in pts]
        ys = [pts[n][1] for n in nodes if n in pts]
        if not xs:
            return (-0.40, 0.02, 0.22)
        cx = float(np.mean(xs))
        cy = float(np.mean(ys))
        r = float(max(np.hypot(np.array(xs) - cx, np.array(ys) - cy)))
        return cx, cy, r

    DE_CX, DE_CY, DE_R = _center_radius(init_pos, DE_nodes)

    # 2) S (ES) fixed outer ring
    S_ORDER = ["GS", "FS", "WECS", "GD", "FD"]
    s_present = [n for n in S_ORDER if n in G]
    R_SCALE, R_MARGIN, S_R_MIN = 1.25, 0.06, 0.26
    S_CX, S_CY = DE_CX, DE_CY
    S_R = max(DE_R * R_SCALE + R_MARGIN, S_R_MIN)
    START_DEG = -90

    if s_present:
        angs = np.deg2rad(np.linspace(START_DEG, START_DEG + 360, len(s_present), endpoint=False))
        s_pos = {name: (S_CX + S_R * np.cos(t), S_CY + S_R * np.sin(t)) for name, t in zip(s_present, angs)}
        init_pos.update(s_pos)
        fixed_nodes.extend(list(s_pos.keys()))
    else:
        s_pos = {}

    # 3) H polar placement outside ES ring
    H_nodes = [n for n in G.nodes() if G.nodes[n].get("layer") == "H"]
    H_MARGIN = 0.18
    if H_nodes and s_pos:
        H_EXTRA = 0.5
        R_H = S_R + H_MARGIN + H_EXTRA
        W_S, W_DE = 1.0, 0.4
        de_pos = {n: init_pos[n] for n in DE_nodes if n in init_pos}

        angles = []
        for h in H_nodes:
            vx = vy = totw = 0.0
            for s in s_pos:
                if G.has_edge(h, s):
                    w = float(G[h][s].get("weight", 0.0)) * W_S
                    if w > 0:
                        sx, sy = s_pos[s]
                        vx += w * (sx - S_CX)
                        vy += w * (sy - S_CY)
                        totw += w
            for de in de_pos:
                if G.has_edge(h, de):
                    w = float(G[h][de].get("weight", 0.0)) * W_DE
                    if w > 0:
                        dx, dy = de_pos[de]
                        vx += w * (dx - S_CX)
                        vy += w * (dy - S_CY)
                        totw += w
            theta = math.atan2(vy, vx) if totw > 0 else math.radians(30.0)
            angles.append([h, theta])

        angles.sort(key=lambda z: z[1])
        DTH_MIN = math.radians(10.0)
        base = angles[0][1]
        angles[0][1] = base
        for i in range(1, len(angles)):
            th = angles[i][1]
            while th < angles[i - 1][1]:
                th += 2 * math.pi
            if th - angles[i - 1][1] < DTH_MIN:
                th = angles[i - 1][1] + DTH_MIN
            angles[i][1] = th

        for name, th in angles:
            while th > math.pi:
                th -= 2 * math.pi
            while th <= -math.pi:
                th += 2 * math.pi
            init_pos[name] = (S_CX + R_H * np.cos(th), S_CY + R_H * np.sin(th))
    else:
        for h in H_nodes:
            init_pos[h] = (S_CX + S_R + H_MARGIN + 0.05, S_CY)

    # 4) spring layout fine-tune (ES fixed)
    G_layout = nx.Graph()
    G_layout.add_nodes_from(G.nodes())
    for u, v, d in G.edges(data=True):
        w = float(d.get("weight", 0.0))
        G_layout.add_edge(u, v, weight=w)

    try:
        pos = nx.spring_layout(
            G_layout,
            weight="weight",
            seed=42,
            pos=init_pos if init_pos else None,
            fixed=fixed_nodes if fixed_nodes else None,
            k=0.6,
            iterations=300,
        )
    except TypeError:
        pos = nx.spring_layout(G_layout, weight="weight", seed=42, pos=init_pos, k=0.6, iterations=300)

    # collision relax (keep your logic)
    ALL_NODES = list(G.nodes())
    FIXED_SET = set(fixed_nodes)
    R_MIN, R_MAX, PADDING = 0.030, 0.060, 0.010
    STEP, MAX_IT, CLIP_XY = 0.55, 80, 0.98

    strength_now = compute_strength(G)
    s_max = max(strength_now.values()) if strength_now else 1.0

    def node_radius(n):
        if s_max <= 0:
            return R_MIN + 0.5 * R_MAX
        return R_MIN + R_MAX * math.sqrt(max(strength_now.get(n, 0.0), 0.0) / s_max)

    R_SOFT_MIN = S_R + H_MARGIN + 0.04
    for _ in range(MAX_IT):
        moved = 0
        for i in range(len(ALL_NODES)):
            ni = ALL_NODES[i]
            if ni not in pos:
                continue
            xi, yi = pos[ni]
            ri = node_radius(ni)
            for j in range(i + 1, len(ALL_NODES)):
                nj = ALL_NODES[j]
                if nj not in pos:
                    continue
                xj, yj = pos[nj]
                rj = node_radius(nj)
                dx, dy = xi - xj, yi - yj
                dist = math.hypot(dx, dy) + 1e-12
                need = ri + rj + PADDING
                if dist < need:
                    ux, uy = dx / dist, dy / dist
                    push = (need - dist) * 0.5 * STEP
                    if ni not in FIXED_SET:
                        xi += ux * push
                        yi += uy * push
                        xi = max(-CLIP_XY, min(CLIP_XY, xi))
                        yi = max(-CLIP_XY, min(CLIP_XY, yi))
                    if nj not in FIXED_SET:
                        xj -= ux * push
                        yj -= uy * push
                        xj = max(-CLIP_XY, min(CLIP_XY, xj))
                        yj = max(-CLIP_XY, min(CLIP_XY, yj))
                    pos[ni] = (xi, yi)
                    pos[nj] = (xj, yj)
                    moved += 1

        for h in H_nodes:
            if h in pos:
                dx, dy = pos[h][0] - S_CX, pos[h][1] - S_CY
                r = math.hypot(dx, dy)
                if r < R_SOFT_MIN:
                    k = R_SOFT_MIN / (r + 1e-12)
                    pos[h] = (S_CX + dx * k, S_CY + dy * k)

        if moved == 0:
            break

    # 5) edges (intra vs inter)
    edges_intra, edges_inter = [], []
    for u, v, d in G.edges(data=True):
        w = float(d.get("weight", 0.0))
        if G.nodes[u].get("layer") == G.nodes[v].get("layer"):
            edges_intra.append((u, v, w))
        else:
            edges_inter.append((u, v, w))

    def _norm_width(ws, base, span):
        if not ws:
            return []
        arr = np.array(ws, float)
        mn, mx = float(arr.min()), float(arr.max())
        if mx <= mn:
            return [base + 0.5 * span] * len(ws)
        return [base + span * ((w - mn) / (mx - mn)) for w in ws]

    if edges_intra:
        wlist = [w for (_, _, w) in edges_intra]
        widths = _norm_width(wlist, base=0.55, span=1.2)
        cols = [(*mpl.colors.to_rgb("#9A9A9A"), 0.45)] * len(edges_intra)
        lc = nx.draw_networkx_edges(
            G, pos, ax=ax,
            edgelist=[(u, v) for (u, v, _) in edges_intra],
            width=widths, edge_color=cols
        )
        if hasattr(lc, "set_zorder"):
            lc.set_zorder(1)

    if edges_inter:
        wlist = [w for (_, _, w) in edges_inter]
        widths = _norm_width(wlist, base=0.80, span=2.2)
        cols = [(*mpl.colors.to_rgb(INTER_YELLOW), 0.45)] * len(edges_inter)
        lc = nx.draw_networkx_edges(
            G, pos, ax=ax,
            edgelist=[(u, v) for (u, v, _) in edges_inter],
            width=widths, edge_color=cols
        )
        if hasattr(lc, "set_zorder"):
            lc.set_zorder(2)

    # 6) nodes
    strength = compute_strength(G)
    s_ser = pd.Series(strength, dtype=float)
    if s_ser.size > 0 and s_ser.max() > 0:
        sizes = (NODE_BASE + NODE_SCALE * np.sqrt(s_ser / s_ser.max())).values
    else:
        sizes = np.full(len(G.nodes()), NODE_BASE + 0.4 * NODE_SCALE)

    node_colors = [COLOR_MAP.get(G.nodes[n].get("layer", "U"), "#7f7f7f") for n in G.nodes()]
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors, node_size=sizes,
        alpha=0.96, linewidths=1.2, edgecolors="white"
    )

    # 7) labels
    labels_dict = {n: n for n in G.nodes()}
    pos_labels = {n: (pos[n][0], pos[n][1] + LABEL_OFFSET) for n in G.nodes() if n in pos}
    texts = nx.draw_networkx_labels(
        G, pos_labels, labels=labels_dict,
        ax=ax, font_size=LABEL_SIZE, font_color="black"
    )
    for t in texts.values():
        t.set_path_effects([pe.withStroke(linewidth=1.5, foreground="white")])

    # 8) legend
    present_raw = {G.nodes[n].get("layer", "U") for n in G.nodes()}
    present = [k for k in LEGEND_ORDER if k in present_raw]

    node_legend = [
        Line2D([0], [0], marker="o", markersize=8, color="none",
               markerfacecolor=COLOR_MAP.get(lyr, "#7f7f7f"),
               label=f"{DISPLAY_CODE.get(lyr,'U')}: {DISPLAY_NAME.get(DISPLAY_CODE.get(lyr,'U'),'Unknown')}")
        for lyr in present
    ]
    edge_legend = [
        Line2D([0, 1], [0, 0], color="#9A9A9A", lw=2.0, label="Intra-layer"),
        Line2D([0, 1], [0, 0], color=INTER_YELLOW, lw=2.4, label="Inter-layer"),
    ]
    handles = node_legend + edge_legend
    n_items = len(handles)
    ncol = int(np.ceil(n_items / 2))

    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.06, top=0.999)
    ax.margins(0, 0)

    band_y = 0.032
    fig.text(0.03, band_y, title_main, ha="left", va="center",
             fontsize=10, fontweight="bold", linespacing=1.15)

    fig.legend(handles=handles,
               loc="center right", bbox_to_anchor=(0.97, band_y),
               ncol=ncol, frameon=False, fontsize=9,
               handlelength=2.0, columnspacing=1.2, borderaxespad=0.0)

    if note_text:
        # keep it optional, and subtle
        fig.text(0.03, band_y - 0.018, str(note_text), ha="left", va="center", fontsize=8)

    plt.savefig(str(png_path), dpi=dpi, bbox_inches="tight", pad_inches=0.01)
    plt.close()


def filter_backbone_topk(df: pd.DataFrame, topk: int):
    """
    Keep top-K edges by NMI.
    If topk <= 0: keep all.
    """
    if topk is None or int(topk) <= 0:
        return df
    df = df.sort_values("nmi", ascending=False)
    return df.head(int(topk))


def main():
    # ---- project-relative IO ----
    SCALE_TAG = "M_IM"
    EDGES_FILE = OUT_ROOT / "network" / f"edges_{SCALE_TAG}_MI_NMI.parquet"

    # ---- choose zone & years ----
    ZONE_KEY = "Mongolia"        # or "Inner_Mongolia"
    YEAR_LIST = None            # None => all years available

    # ---- backbone control ----
    TOPK = 60                   # set 0 to plot full dense network

    # ---- output ----
    FIG_ROOT = OUT_ROOT / "network" / "fig_net" / SCALE_TAG / ZONE_KEY
    FIG_ROOT.mkdir(parents=True, exist_ok=True)

    if not EDGES_FILE.exists():
        raise SystemExit(f"Edges file not found: {EDGES_FILE}")

    df = pd.read_parquet(EDGES_FILE)
    if df.empty:
        raise SystemExit("Edges table is empty.")

    # basic checks
    for c in ["zone_key", "time", "var_i", "var_j", "nmi"]:
        if c not in df.columns:
            raise SystemExit(f"Missing column in edges table: {c}")

    df["time"] = pd.to_datetime(df["time"])
    df = df[df["zone_key"].astype(str) == str(ZONE_KEY)].copy()
    if df.empty:
        raise SystemExit(f"No records for zone_key='{ZONE_KEY}'")

    years_all = sorted(df["time"].dt.year.unique().tolist())
    if YEAR_LIST is None:
        years = years_all
    else:
        years = [int(y) for y in YEAR_LIST if int(y) in years_all]

    if not years:
        raise SystemExit("No years to plot.")

    stamp(f"zone={ZONE_KEY} | years={len(years)} | TOPK={TOPK}")

    for yr in years:
        dyy = df[df["time"].dt.year == int(yr)].copy()
        if dyy.empty:
            continue

        dyy = filter_backbone_topk(dyy, TOPK)
        G = build_graph_from_edges(dyy)

        N, M = G.number_of_nodes(), G.number_of_edges()
        if ZONE_KEY == "Inner_Mongolia":
            title_main = f"     {yr}\n   IM-SENs"
        else:
            title_main = f"     {yr}\n   M-SENs"

        note = f"nodes={N} edges={M}  (MI/NMI, topK={TOPK})"

        out_png = FIG_ROOT / f"net_{ZONE_KEY}_{yr}.png"
        draw_graph(G, title_main=title_main, png_path=out_png, note_text=note)
        stamp(f"saved: {out_png}")

    stamp("All done.")


if __name__ == "__main__":
    main()
