"""
buckets.py

Analysis of critical positions in "buckets" made from the square on which
the piece stood that had to make the best move:

    piece type   pawn / knight / bishop / rook / queen / king
    file         a b c d e f g h        (left -> right as seen by White)
    half         "own half"             (ranks 1-4 for White, 5-8 for Black)
                 "opponent half"        (the other four ranks)
    in check?    "defence against check" (the side to move is in check)
                 "no check"

Input is the natural (NOT balanced) data - e.g. `critical_positions.csv`
made by find_critical_positions: every analysed position, with columns
`best_move`, `best_piece`, `side_to_move`, `is_critical`, `found_best`,
`player_elo`. The number of positions in every bucket is therefore the
real one.

For each bucket we report:
    n_positions     how many analysed positions had their best move in the bucket
    n_critical      how many of them were critical
    critical_share  n_critical / n_positions
    n_found         critical positions where the player found the best move
    found_rate      n_found / n_critical   (+ 95% Wilson interval)
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

FILES = list("abcdefgh")
PIECE_ORDER = ["pawn", "knight", "bishop", "rook", "queen", "king"]
OWN, OPP = "own half", "opponent half"
HALF_ORDER = [OWN, OPP]
CHECK, NOCHECK = "defence against check", "no check"
CHECK_ORDER = [CHECK, NOCHECK]


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def add_bucket_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add best_file (a-h), best_rank (1-8), rel_rank (1-8 from the point of
    view of the player to move) and half (own / opponent) - all from the
    square where the piece making the BEST move stood."""
    out = df.copy()
    uci = out["best_move"].astype(str)
    out["best_file"] = uci.str[0]
    out["best_rank"] = uci.str[1].astype(int)
    white = out["side_to_move"].eq("white")
    out["rel_rank"] = out["best_rank"].where(white, 9 - out["best_rank"])
    out["half"] = np.where(out["rel_rank"] <= 4, OWN, OPP)
    out["is_critical"] = out["is_critical"].astype(str).str.lower().eq("true")
    out["found_best"] = out["found_best"].astype(str).str.lower().eq("true")
    if "fen" in out.columns:
        import chess                       # only needed to read the FEN
        out["in_check"] = [chess.Board(f).is_check() for f in out["fen"]]
        out["check_status"] = pd.Categorical(
            np.where(out["in_check"], CHECK, NOCHECK), categories=CHECK_ORDER, ordered=True)
    out["best_piece"] = pd.Categorical(out["best_piece"], categories=PIECE_ORDER, ordered=True)
    out["best_file"] = pd.Categorical(out["best_file"], categories=FILES, ordered=True)
    out["half"] = pd.Categorical(out["half"], categories=HALF_ORDER, ordered=True)
    return out


def wilson(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion k/n."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarize(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """One row per bucket (combination of the `by` columns)."""
    d = df.copy()
    d["_found_crit"] = d["found_best"] & d["is_critical"]
    g = d.groupby(by, observed=True).agg(
        n_positions=("is_critical", "size"),
        n_critical=("is_critical", "sum"),
        n_found=("_found_crit", "sum"),
    ).reset_index()
    g["critical_share"] = g["n_critical"] / g["n_positions"]
    g["found_rate"] = g["n_found"] / g["n_critical"].where(g["n_critical"] > 0)
    ci = [wilson(k, n) for k, n in zip(g["n_found"], g["n_critical"])]
    g["ci_low"] = [c[0] for c in ci]
    g["ci_high"] = [c[1] for c in ci]
    return g


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_piece_overview(df: pd.DataFrame):
    """Per piece: how many critical positions (real distribution) and how
    often the best move was found, with 95% interval."""
    t = summarize(df, ["best_piece"])
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    x = np.arange(len(t))
    ax[0].bar(x, t["n_positions"], color="#b8c4d6", label="all positions")
    ax[0].bar(x, t["n_critical"], color="#3b6ea5", label="critical")
    for i, (a, b) in enumerate(zip(t["n_positions"], t["n_critical"])):
        ax[0].text(i, a, f"{a}\n({b} critical)", ha="center", va="bottom", fontsize=8)
    ax[0].set_title("Number of positions, by the piece making the best move")
    ax[0].legend(fontsize=8)
    ax[1].bar(x, t["critical_share"], color="#c98b2b")
    for i, v in enumerate(t["critical_share"]):
        ax[1].text(i, v, f"{v:.1%}", ha="center", va="bottom", fontsize=8)
    ax[1].set_title("Share of positions that are critical")
    err = np.vstack([t["found_rate"] - t["ci_low"], t["ci_high"] - t["found_rate"]])
    ax[2].bar(x, t["found_rate"], yerr=np.nan_to_num(err), capsize=4, color="#4c9a6a")
    for i, (v, n) in enumerate(zip(t["found_rate"], t["n_critical"])):
        ax[2].text(i, (0 if np.isnan(v) else v) + 0.02, f"{v:.0%}\nn={n}", ha="center", va="bottom", fontsize=8)
    ax[2].set_ylim(0, 1.15)
    ax[2].set_title("How often the best move was found (95% CI)")
    for a in ax:
        a.set_xticks(x)
        a.set_xticklabels(t["best_piece"].astype(str))
    fig.tight_layout()
    return fig


def plot_half(df: pd.DataFrame):
    """Found rate per piece, own half vs opponent's half."""
    t = summarize(df, ["best_piece", "half"])
    fig, ax = plt.subplots(figsize=(11, 4.5))
    width = 0.38
    colors = {OWN: "#3b6ea5", OPP: "#d1704b"}
    for j, half in enumerate(HALF_ORDER):
        sub = t[t["half"] == half].set_index("best_piece").reindex(PIECE_ORDER)
        x = np.arange(len(PIECE_ORDER)) + (j - 0.5) * width
        rate = sub["found_rate"].to_numpy(dtype=float)
        err = np.vstack([rate - sub["ci_low"].to_numpy(dtype=float),
                         sub["ci_high"].to_numpy(dtype=float) - rate])
        ax.bar(x, np.nan_to_num(rate), width, yerr=np.nan_to_num(err), capsize=3,
               color=colors[half], label=half)
        for xi, v, n in zip(x, rate, sub["n_critical"].fillna(0).astype(int)):
            ax.text(xi, (0 if np.isnan(v) else v) + 0.03, f"{0 if np.isnan(v) else v:.0%}\nn={n}",
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(np.arange(len(PIECE_ORDER)))
    ax.set_xticklabels(PIECE_ORDER)
    ax.set_ylim(0, 1.2)
    ax.set_title("Share of best moves found: own half vs opponent half")
    ax.legend()
    fig.tight_layout()
    return fig


def _annotated_heatmap(ax, rate: pd.DataFrame, n: pd.DataFrame, min_n: int, title: str, show_cbar=False):
    masked = rate.where(n >= min_n)
    sns.heatmap(masked, ax=ax, vmin=0, vmax=1, cmap="RdYlGn", cbar=show_cbar,
                linewidths=0.5, linecolor="white", annot=False)
    ax.set_facecolor("#e6e6e6")                       # cells with too little data
    for i in range(rate.shape[0]):
        for j in range(rate.shape[1]):
            nv = n.iloc[i, j]
            nv = 0 if pd.isna(nv) else int(nv)
            r = rate.iloc[i, j]
            if nv >= min_n and not pd.isna(r):
                ax.text(j + 0.5, i + 0.5, f"{r:.0%}\nn={nv}", ha="center", va="center", fontsize=7)
            elif nv > 0:
                ax.text(j + 0.5, i + 0.5, f"n={nv}", ha="center", va="center", fontsize=6, color="#777777")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("")
    ax.set_ylabel("")


def plot_piece_file(df: pd.DataFrame, min_n: int = 10):
    """Heatmap piece x file (a-h): found rate, annotated with n."""
    t = summarize(df, ["best_piece", "best_file"])
    rate = t.pivot(index="best_piece", columns="best_file", values="found_rate").reindex(index=PIECE_ORDER, columns=FILES)
    n = t.pivot(index="best_piece", columns="best_file", values="n_critical").reindex(index=PIECE_ORDER, columns=FILES)
    fig, ax = plt.subplots(figsize=(11, 4))
    _annotated_heatmap(ax, rate, n, min_n,
                       f"Share of best moves found: piece × file (grey: n < {min_n})", show_cbar=True)
    fig.tight_layout()
    return fig


def plot_full_buckets(df: pd.DataFrame, min_n: int = 10):
    """All buckets at once: one panel per piece, rows = half of the board,
    columns = file a-h. Colour = found rate, text = rate and n (critical)."""
    t = summarize(df, ["best_piece", "half", "best_file"])
    fig, axes = plt.subplots(3, 2, figsize=(13, 10))
    for ax, piece in zip(axes.ravel(), PIECE_ORDER):
        sub = t[t["best_piece"] == piece]
        rate = sub.pivot(index="half", columns="best_file", values="found_rate").reindex(index=HALF_ORDER, columns=FILES)
        n = sub.pivot(index="half", columns="best_file", values="n_critical").reindex(index=HALF_ORDER, columns=FILES)
        _annotated_heatmap(ax, rate, n, min_n, f"{piece}  (grey: n < {min_n})")
    fig.suptitle("Decision accuracy in critical positions: piece × file × half of the board", fontsize=12)
    fig.tight_layout()
    return fig


def plot_piece_file_by_half(df: pd.DataFrame, min_n: int = 10):
    """Two big heatmaps side by side: OWN half | OPPONENT's half.
    Rows = piece, columns = file a-h, text = found rate and n."""
    t = summarize(df, ["best_piece", "half", "best_file"])
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5), sharey=True)
    for k, (ax, half) in enumerate(zip(axes, HALF_ORDER)):
        sub = t[t["half"] == half]
        rate = sub.pivot(index="best_piece", columns="best_file", values="found_rate").reindex(index=PIECE_ORDER, columns=FILES)
        n = sub.pivot(index="best_piece", columns="best_file", values="n_critical").reindex(index=PIECE_ORDER, columns=FILES)
        _annotated_heatmap(ax, rate, n, min_n, f"{half.upper()}  (grey: n < {min_n})", show_cbar=(k == 1))
    fig.suptitle("Share of best moves found: piece × file, own half and opponent half shown separately", fontsize=12)
    fig.tight_layout()
    return fig


def plot_piece_elo(df: pd.DataFrame, bins: list[int], labels: list[str], min_n: int = 10):
    """Found rate per piece and Elo bin (bins can be changed freely)."""
    d = df.copy()
    d["elo_bin"] = pd.cut(d["player_elo"], bins=bins, labels=labels)
    t = summarize(d, ["best_piece", "elo_bin"])
    rate = t.pivot(index="best_piece", columns="elo_bin", values="found_rate").reindex(index=PIECE_ORDER, columns=labels)
    n = t.pivot(index="best_piece", columns="elo_bin", values="n_critical").reindex(index=PIECE_ORDER, columns=labels)
    fig, ax = plt.subplots(figsize=(8, 4))
    _annotated_heatmap(ax, rate, n, min_n,
                       f"Share of best moves found: piece × Elo (grey: n < {min_n})", show_cbar=True)
    fig.tight_layout()
    return fig


def bucket_table(df: pd.DataFrame, with_check: bool = True) -> pd.DataFrame:
    """The full table: one row per (piece, half, file) bucket, additionally
    split into 'defence against check' / 'no check' when with_check=True."""
    by = ["best_piece", "half", "best_file"]
    if with_check and "check_status" in df.columns:
        by = by + ["check_status"]
    t = summarize(df, by)
    return t.sort_values(by).reset_index(drop=True)


def plot_check(df: pd.DataFrame):
    """Defence against check (the side to move is in check) vs. other
    positions: how common, how often critical, how often the best move is
    found - overall and per piece that makes the best move."""
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    colors = {CHECK: "#c0392b", NOCHECK: "#7f8c8d"}

    t = summarize(df, ["check_status"]).set_index("check_status").reindex(CHECK_ORDER)
    x = np.arange(2)
    ax[0].bar(x, t["n_positions"], color=[colors[c] for c in CHECK_ORDER])
    for i, (a, b) in enumerate(zip(t["n_positions"], t["n_critical"])):
        ax[0].text(i, a, f"{a}\n({b} critical, {b / a:.1%})", ha="center", va="bottom", fontsize=8)
    ax[0].set_ylim(0, float(t["n_positions"].max()) * 1.2)
    ax[0].set_title("Number of positions and critical positions")

    rate = t["found_rate"].to_numpy(dtype=float)
    err = np.vstack([rate - t["ci_low"].to_numpy(dtype=float), t["ci_high"].to_numpy(dtype=float) - rate])
    ax[1].bar(x, np.nan_to_num(rate), yerr=np.nan_to_num(err), capsize=4, color=[colors[c] for c in CHECK_ORDER])
    for i, (v, n) in enumerate(zip(rate, t["n_critical"])):
        ax[1].text(i, (0 if np.isnan(v) else v) + 0.03, f"{0 if np.isnan(v) else v:.0%}\nn={n}", ha="center", va="bottom", fontsize=8)
    ax[1].set_ylim(0, 1.2)
    ax[1].set_title("Share of best moves found (95% CI)")
    for a in ax[:2]:
        a.set_xticks(x)
        a.set_xticklabels(CHECK_ORDER)

    t2 = summarize(df, ["best_piece", "check_status"])
    width = 0.38
    for j, cs in enumerate(CHECK_ORDER):
        sub = t2[t2["check_status"] == cs].set_index("best_piece").reindex(PIECE_ORDER)
        xs = np.arange(len(PIECE_ORDER)) + (j - 0.5) * width
        r = sub["found_rate"].to_numpy(dtype=float)
        e = np.vstack([r - sub["ci_low"].to_numpy(dtype=float), sub["ci_high"].to_numpy(dtype=float) - r])
        ax[2].bar(xs, np.nan_to_num(r), width, yerr=np.nan_to_num(e), capsize=2, color=colors[cs], label=cs)
        for xi, v, n in zip(xs, r, sub["n_critical"].fillna(0).astype(int)):
            ax[2].text(xi, (0 if np.isnan(v) else v) + 0.03, f"n={n}", ha="center", va="bottom", fontsize=6, rotation=90)
    ax[2].set_xticks(np.arange(len(PIECE_ORDER)))
    ax[2].set_xticklabels(PIECE_ORDER)
    ax[2].set_ylim(0, 1.3)
    ax[2].set_title("By the piece making the best move")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_check_elo(df: pd.DataFrame, bins: list[int], labels: list[str]):
    """Found rate by Elo bin: defence against check vs. other positions."""
    d = df.copy()
    d["elo_bin"] = pd.cut(d["player_elo"], bins=bins, labels=labels)
    t = summarize(d, ["check_status", "elo_bin"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = {CHECK: "#c0392b", NOCHECK: "#7f8c8d"}
    for j, cs in enumerate(CHECK_ORDER):
        sub = t[t["check_status"] == cs].set_index("elo_bin").reindex(labels)
        xs = np.arange(len(labels)) + (j - 0.5) * 0.1
        r = sub["found_rate"].to_numpy(dtype=float)
        e = np.vstack([r - sub["ci_low"].to_numpy(dtype=float), sub["ci_high"].to_numpy(dtype=float) - r])
        ax.errorbar(xs, r, yerr=np.nan_to_num(e), marker="o", capsize=3, color=colors[cs], label=cs)
        for xi, v, n in zip(xs, r, sub["n_critical"].fillna(0).astype(int)):
            if not np.isnan(v):
                ax.text(xi, v + 0.04, f"n={n}", ha="center", fontsize=7, color=colors[cs])
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    ax.set_title("Share of best moves found by Elo: defence against check vs the rest")
    ax.legend()
    fig.tight_layout()
    return fig
