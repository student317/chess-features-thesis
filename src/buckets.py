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
`player_elo`.

For each bucket we report:
    n_positions     how many analysed positions had their best move in the bucket
    n_critical      how many of them were critical
    critical_share  n_critical / n_positions
    n_found         critical positions where the player found the best move
    blunder_rate    1 - n_found / n_critical   (+ 95% Wilson interval)   <- used in all plots
    found_rate      n_found / n_critical       (+ ci_low / ci_high)      <- kept for compatibility

All plots show the blunder rate (higher = players fail more often) with its
95% confidence interval.
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
CMAP = "RdYlGn_r"          # green = few blunders, red = many


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


def wilson(k: float, n: float, z: float = 1.96) -> tuple:
    """95% Wilson score interval for a proportion k/n."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def add_blunder_columns(g: pd.DataFrame, k_col: str = "n_found", n_col: str = "n_critical") -> pd.DataFrame:
    """Add found_rate / blunder_rate and their 95% intervals to a table with
    `n_found` (successes) and `n_critical` (trials)."""
    n = g[n_col].where(g[n_col] > 0)
    g["found_rate"] = g[k_col] / n
    g["blunder_rate"] = 1 - g["found_rate"]
    ci = [wilson(k, m) for k, m in zip(g[k_col], g[n_col])]
    g["ci_low"] = [c[0] for c in ci]
    g["ci_high"] = [c[1] for c in ci]
    g["blunder_ci_low"] = 1 - g["ci_high"]
    g["blunder_ci_high"] = 1 - g["ci_low"]
    return g


def summarize(df: pd.DataFrame, by: list) -> pd.DataFrame:
    """One row per bucket (combination of the `by` columns)."""
    d = df.copy()
    d["_found_crit"] = d["found_best"] & d["is_critical"]
    g = d.groupby(by, observed=True).agg(
        n_positions=("is_critical", "size"),
        n_critical=("is_critical", "sum"),
        n_found=("_found_crit", "sum"),
    ).reset_index()
    g["critical_share"] = g["n_critical"] / g["n_positions"]
    return add_blunder_columns(g)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _pivots(t, index, columns, idx_order, col_order):
    def piv(v):
        return t.pivot(index=index, columns=columns, values=v).reindex(index=idx_order, columns=col_order)
    return piv("blunder_rate"), piv("n_critical"), piv("blunder_ci_low"), piv("blunder_ci_high")


def _err(rate, lo, hi):
    rate, lo, hi = (np.asarray(x, dtype=float) for x in (rate, lo, hi))
    # clip: for rates of exactly 0 or 1 rounding can give -1e-16, matplotlib rejects negative bars
    return np.clip(np.nan_to_num(np.vstack([rate - lo, hi - rate])), 0, None)


def _annotated_heatmap(ax, rate, n, min_n, title, show_cbar=False, lo=None, hi=None, fontsize=7,
                       vmax=1.0, cmap=CMAP, cbar_label="blunder rate"):
    """Heatmap of blunder rates. Text: rate, [95% CI] and n. Cells with
    fewer than min_n critical positions are grey and show only n."""
    masked = rate.where(n >= min_n)
    sns.heatmap(masked, ax=ax, vmin=0, vmax=vmax, cmap=cmap, cbar=show_cbar,
                cbar_kws={"label": cbar_label} if show_cbar else None,
                linewidths=0.5, linecolor="white", annot=False)
    ax.set_facecolor("#e6e6e6")
    for i in range(rate.shape[0]):
        for j in range(rate.shape[1]):
            nv = n.iloc[i, j]
            nv = 0 if pd.isna(nv) else int(nv)
            r = rate.iloc[i, j]
            if nv >= min_n and not pd.isna(r):
                if lo is not None and hi is not None:
                    txt = f"{r:.0%}\n[{lo.iloc[i, j]:.0%}-{hi.iloc[i, j]:.0%}]\nn={nv}"
                else:
                    txt = f"{r:.0%}\nn={nv}"
                ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center", fontsize=fontsize)
            elif nv > 0:
                ax.text(j + 0.5, i + 0.5, f"n={nv}", ha="center", va="center", fontsize=fontsize - 1, color="#777777")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("")
    ax.set_ylabel("")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_piece_overview(df: pd.DataFrame):
    """Per piece: how many critical positions (real distribution), what
    share of positions is critical, and the blunder rate with 95% CI."""
    t = summarize(df, ["best_piece"])
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    x = np.arange(len(t))
    ax[0].bar(x, t["n_positions"], color="#b8c4d6", label="all positions")
    ax[0].bar(x, t["n_critical"], color="#3b6ea5", label="critical")
    for i, (a, b) in enumerate(zip(t["n_positions"], t["n_critical"])):
        ax[0].text(i, a, f"{a}\n({b} critical)", ha="center", va="bottom", fontsize=8)
    ax[0].set_ylim(0, float(t["n_positions"].max()) * 1.2)
    ax[0].set_title("Number of positions, by the piece making the best move")
    ax[0].legend(fontsize=8)
    # share of positions that are critical, with CI
    ci = [wilson(k, n) for k, n in zip(t["n_critical"], t["n_positions"])]
    sh = t["critical_share"].to_numpy(dtype=float)
    ax[1].bar(x, sh, yerr=_err(sh, [c[0] for c in ci], [c[1] for c in ci]), capsize=4, color="#c98b2b")
    for i, v in enumerate(sh):
        ax[1].text(i, v + 0.002, f"{v:.1%}", ha="center", va="bottom", fontsize=8)
    ax[1].set_title("Share of positions that are critical (95% CI)")
    br = t["blunder_rate"].to_numpy(dtype=float)
    ax[2].bar(x, np.nan_to_num(br), yerr=_err(br, t["blunder_ci_low"], t["blunder_ci_high"]), capsize=4, color="#c0504d")
    for i, (v, n) in enumerate(zip(br, t["n_critical"])):
        ax[2].text(i, (0 if np.isnan(v) else v) + 0.03, f"{0 if np.isnan(v) else v:.0%}\nn={n}", ha="center", va="bottom", fontsize=8)
    ax[2].set_ylim(0, 1.2)
    ax[2].set_title("Blunder rate in critical positions (95% CI)")
    for a in ax:
        a.set_xticks(x)
        a.set_xticklabels(t["best_piece"].astype(str))
    fig.tight_layout()
    return fig


def plot_half(df: pd.DataFrame):
    """Blunder rate per piece, own half vs opponent half."""
    t = summarize(df, ["best_piece", "half"])
    fig, ax = plt.subplots(figsize=(11, 4.5))
    width = 0.38
    colors = {OWN: "#3b6ea5", OPP: "#d1704b"}
    for j, half in enumerate(HALF_ORDER):
        sub = t[t["half"] == half].set_index("best_piece").reindex(PIECE_ORDER)
        x = np.arange(len(PIECE_ORDER)) + (j - 0.5) * width
        rate = sub["blunder_rate"].to_numpy(dtype=float)
        ax.bar(x, np.nan_to_num(rate), width, yerr=_err(rate, sub["blunder_ci_low"], sub["blunder_ci_high"]),
               capsize=3, color=colors[half], label=half)
        for xi, v, n in zip(x, rate, sub["n_critical"].fillna(0).astype(int)):
            ax.text(xi, (0 if np.isnan(v) else v) + 0.03, f"{0 if np.isnan(v) else v:.0%}\nn={n}",
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(np.arange(len(PIECE_ORDER)))
    ax.set_xticklabels(PIECE_ORDER)
    ax.set_ylim(0, 1.25)
    ax.set_title("Blunder rate in critical positions: own half vs opponent half (95% CI)")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_piece_file(df: pd.DataFrame, min_n: int = 10):
    """Heatmap piece x file (a-h): blunder rate with CI and n."""
    t = summarize(df, ["best_piece", "best_file"])
    rate, n, lo, hi = _pivots(t, "best_piece", "best_file", PIECE_ORDER, FILES)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    _annotated_heatmap(ax, rate, n, min_n, f"Blunder rate: piece x file (grey: n < {min_n})", show_cbar=True, lo=lo, hi=hi, fontsize=6.5)
    fig.tight_layout()
    return fig


def plot_full_buckets(df: pd.DataFrame, min_n: int = 10):
    """All buckets at once: one panel per piece, rows = half of the board,
    columns = file a-h. Colour = blunder rate, text = rate, CI and n."""
    t = summarize(df, ["best_piece", "half", "best_file"])
    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    for ax, piece in zip(axes.ravel(), PIECE_ORDER):
        sub = t[t["best_piece"] == piece]
        rate, n, lo, hi = _pivots(sub, "half", "best_file", HALF_ORDER, FILES)
        _annotated_heatmap(ax, rate, n, min_n, f"{piece}  (grey: n < {min_n})", lo=lo, hi=hi, fontsize=5.5)
    fig.suptitle("Blunder rate in critical positions: piece x file x half of the board", fontsize=12)
    fig.tight_layout()
    return fig


def plot_piece_file_by_half(df: pd.DataFrame, min_n: int = 10):
    """Two big heatmaps side by side: OWN half | OPPONENT's half.
    Rows = piece, columns = file a-h, text = blunder rate, CI and n."""
    t = summarize(df, ["best_piece", "half", "best_file"])
    fig, axes = plt.subplots(1, 2, figsize=(16, 4.8), sharey=True)
    for k, (ax, half) in enumerate(zip(axes, HALF_ORDER)):
        rate, n, lo, hi = _pivots(t[t["half"] == half], "best_piece", "best_file", PIECE_ORDER, FILES)
        _annotated_heatmap(ax, rate, n, min_n, f"{half.upper()}  (grey: n < {min_n})",
                           show_cbar=(k == 1), lo=lo, hi=hi, fontsize=6)
    fig.suptitle("Blunder rate: piece x file, own half and opponent half shown separately", fontsize=12)
    fig.tight_layout()
    return fig


def plot_piece_elo(df: pd.DataFrame, bins: list, labels: list, min_n: int = 10):
    """Blunder rate per piece and Elo bin (bins can be changed freely)."""
    d = df.copy()
    d["elo_bin"] = pd.cut(d["player_elo"], bins=bins, labels=labels)
    t = summarize(d, ["best_piece", "elo_bin"])
    rate, n, lo, hi = _pivots(t, "best_piece", "elo_bin", PIECE_ORDER, labels)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    _annotated_heatmap(ax, rate, n, min_n, f"Blunder rate: piece x Elo (grey: n < {min_n})", show_cbar=True, lo=lo, hi=hi)
    fig.tight_layout()
    return fig


def bucket_table(df: pd.DataFrame, with_check: bool = True) -> pd.DataFrame:
    """The full table: one row per (piece, half, file) bucket, additionally
    split into 'defence against check' / 'no check' when with_check=True.
    Contains blunder_rate and its 95% interval."""
    by = ["best_piece", "half", "best_file"]
    if with_check and "check_status" in df.columns:
        by = by + ["check_status"]
    t = summarize(df, by)
    return t.sort_values(by).reset_index(drop=True)


def plot_check(df: pd.DataFrame):
    """Defence against check (the side to move is in check) vs. other
    positions: how common, how often critical, blunder rate - overall and
    per piece that makes the best move. 95% CI everywhere."""
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    colors = {CHECK: "#c0392b", NOCHECK: "#7f8c8d"}

    t = summarize(df, ["check_status"]).set_index("check_status").reindex(CHECK_ORDER)
    x = np.arange(2)
    ax[0].bar(x, t["n_positions"], color=[colors[c] for c in CHECK_ORDER])
    for i, (a, b) in enumerate(zip(t["n_positions"], t["n_critical"])):
        lo, hi = wilson(b, a)
        ax[0].text(i, a, f"{a}\n({b} critical, {b / a:.1%}\n[{lo:.1%}-{hi:.1%}])", ha="center", va="bottom", fontsize=8)
    ax[0].set_ylim(0, float(t["n_positions"].max()) * 1.3)
    ax[0].set_title("Number of positions and critical positions")

    rate = t["blunder_rate"].to_numpy(dtype=float)
    ax[1].bar(x, np.nan_to_num(rate), yerr=_err(rate, t["blunder_ci_low"], t["blunder_ci_high"]), capsize=4,
              color=[colors[c] for c in CHECK_ORDER])
    for i, (v, n) in enumerate(zip(rate, t["n_critical"])):
        ax[1].text(i, (0 if np.isnan(v) else v) + 0.03, f"{0 if np.isnan(v) else v:.0%}\nn={n}", ha="center", va="bottom", fontsize=8)
    ax[1].set_ylim(0, 1.2)
    ax[1].set_title("Blunder rate (95% CI)")
    for a in ax[:2]:
        a.set_xticks(x)
        a.set_xticklabels(CHECK_ORDER)

    t2 = summarize(df, ["best_piece", "check_status"])
    width = 0.38
    for j, cs in enumerate(CHECK_ORDER):
        sub = t2[t2["check_status"] == cs].set_index("best_piece").reindex(PIECE_ORDER)
        xs = np.arange(len(PIECE_ORDER)) + (j - 0.5) * width
        r = sub["blunder_rate"].to_numpy(dtype=float)
        ax[2].bar(xs, np.nan_to_num(r), width, yerr=_err(r, sub["blunder_ci_low"], sub["blunder_ci_high"]),
                  capsize=2, color=colors[cs], label=cs)
        for xi, v, n in zip(xs, r, sub["n_critical"].fillna(0).astype(int)):
            ax[2].text(xi, (0 if np.isnan(v) else v) + 0.03, f"n={n}", ha="center", va="bottom", fontsize=6, rotation=90)
    ax[2].set_xticks(np.arange(len(PIECE_ORDER)))
    ax[2].set_xticklabels(PIECE_ORDER)
    ax[2].set_ylim(0, 1.35)
    ax[2].set_title("By the piece making the best move")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_check_elo(df: pd.DataFrame, bins: list, labels: list):
    """Blunder rate by Elo bin: defence against check vs. other positions."""
    d = df.copy()
    d["elo_bin"] = pd.cut(d["player_elo"], bins=bins, labels=labels)
    t = summarize(d, ["check_status", "elo_bin"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = {CHECK: "#c0392b", NOCHECK: "#7f8c8d"}
    for j, cs in enumerate(CHECK_ORDER):
        sub = t[t["check_status"] == cs].set_index("elo_bin").reindex(labels)
        xs = np.arange(len(labels)) + (j - 0.5) * 0.1
        r = sub["blunder_rate"].to_numpy(dtype=float)
        ax.errorbar(xs, r, yerr=_err(r, sub["blunder_ci_low"], sub["blunder_ci_high"]), marker="o", capsize=3,
                    color=colors[cs], label=cs)
        for xi, v, n in zip(xs, r, sub["n_critical"].fillna(0).astype(int)):
            if not np.isnan(v):
                ax.text(xi, v + 0.04, f"n={n}", ha="center", fontsize=7, color=colors[cs])
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    ax.set_title("Blunder rate by Elo: defence against check vs the rest (95% CI)")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_error_confusion(df: pd.DataFrame, min_n: int = 10):
    """Mistakes in critical positions: which piece did the player move
    instead? Rows = piece that should have moved (best move), columns =
    piece actually moved. Each row sums to 100% (share of that piece's
    mistakes); text = share, 95% CI and number of mistakes. The diagonal
    means the right piece was moved but with the wrong move. Rows with
    fewer than `min_n` mistakes are greyed out."""
    err = df[df["is_critical"] & ~df["found_best"]]
    ct = pd.crosstab(err["best_piece"].astype(str), err["played_piece"].astype(str))
    ct = ct.reindex(index=PIECE_ORDER, columns=PIECE_ORDER, fill_value=0)
    totals = ct.sum(axis=1)
    share = ct.div(totals.where(totals > 0), axis=0)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.heatmap(share.where(totals >= min_n), ax=ax, vmin=0, vmax=1, cmap="Blues",
                cbar_kws={"label": "share of the row"}, linewidths=0.5, linecolor="white")
    ax.set_facecolor("#e6e6e6")
    for i in range(len(PIECE_ORDER)):
        for j in range(len(PIECE_ORDER)):
            n = int(ct.iloc[i, j])
            if totals.iloc[i] >= min_n:
                v = share.iloc[i, j]
                lo, hi = wilson(n, totals.iloc[i])
                ax.text(j + 0.5, i + 0.5, f"{v:.0%}\n[{lo:.0%}-{hi:.0%}]\nn={n}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.5 else "black")
            elif n > 0:
                ax.text(j + 0.5, i + 0.5, f"n={n}", ha="center", va="center", fontsize=6, color="#777777")
    ax.set_xlabel("piece actually moved")
    ax.set_ylabel("piece that should have moved (best move)")
    ax.set_title(f"Mistakes in critical positions: which piece was moved instead (grey rows: < {min_n} mistakes)")
    fig.tight_layout()
    return fig
