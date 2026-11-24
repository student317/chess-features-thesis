"""
tactic_report.py

Statistics on top of enrich.py / tactic_stats.py (nothing there is changed):

 1. co-occurrence of tactics in one move (matrix + table of combinations)
 2. tactic_length: distribution and blunder rate by length (and by tactic)
 3. tactics in unforced errors (missed / allowed) vs. mistakes in critical positions
 4. coverage: how many critical positions / blunders have at least one label
    (target of the supervisor: ~80% of blunders classified)
 5. full_report(...): everything saved to a folder (CSV tables + PNG plots)

Every proportion has a 95% Wilson confidence interval.

Typical use (notebook):

    import enrich, timecontrol
    timecontrol.tag_time_control(PGN, MAIN_CSV)          # once
    enrich.enrich_critical(MAIN_CSV, ENR_CSV, SF, increment=3, compute_length=True)
    enrich.find_unforced_errors(MAIN_CSV, UNF_CSV, SF, increment=3, sample_n=20000)

    import tactic_report as R
    res = R.full_report(MAIN_CSV, ENR_CSV, UNF_CSV, REG_CSV, BINS, LABELS, out_dir="../report")
"""

from __future__ import annotations

import itertools
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from buckets import _err, add_blunder_columns, wilson
from tactic_stats import (blunder_by_tactic, classified_table, explode_labels, load_tactic_data,
                          plot_classified_share, plot_share_heatmap, plot_tactic_blunder_rates,
                          plot_tactic_elo_heatmap, plot_unforced_rate, share_table,
                          unforced_rate_table, with_elo_bins)


def _truthy(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().eq("true")


def _split(s) -> list:
    return [t for t in str(s if isinstance(s, str) else "").split(";") if t]


def _with_ci(df: pd.DataFrame, k: str, n: str, prefix: str = "") -> pd.DataFrame:
    ci = [wilson(a, b) if b > 0 else (np.nan, np.nan) for a, b in zip(df[k], df[n])]
    df[prefix + "ci_low"] = [c[0] for c in ci]
    df[prefix + "ci_high"] = [c[1] for c in ci]
    return df


# ---------------------------------------------------------------------------
# 1. Co-occurrence of tactics in one move
# ---------------------------------------------------------------------------

def combination_table(df: pd.DataFrame, col: str = "missed_tactics", min_n: int = 1) -> pd.DataFrame:
    """One row per exact SET of tactics found in a move (e.g. 'DOUBLE_ATTACK;PIN').
    n, share of all positions (95% CI) and - for the 'missed' view - the blunder rate
    of positions with exactly that combination."""
    d = df.copy()
    d["combo"] = d[col].map(lambda s: ";".join(sorted(_split(s))) or "none")
    d["n_tactics"] = d[col].map(lambda s: len(_split(s)))
    g = d.groupby("combo").agg(n=("found_best", "size"), n_found=("found_best", "sum"),
                               n_tactics=("n_tactics", "first")).reset_index()
    g["share"] = g["n"] / len(d)
    g = _with_ci(g.assign(total=len(d)), "n", "total", "share_")
    g = g.drop(columns="total")
    g["blunder_rate"] = 1 - g["n_found"] / g["n"]
    ci = [wilson(int(n - f), int(n)) for n, f in zip(g["n"], g["n_found"])]
    g["blunder_ci_low"] = [c[0] for c in ci]
    g["blunder_ci_high"] = [c[1] for c in ci]
    return g[g["n"] >= min_n].sort_values("n", ascending=False).reset_index(drop=True)


def cooccurrence_matrix(df: pd.DataFrame, col: str = "missed_tactics") -> pd.DataFrame:
    """Square matrix: cell (A, B) = number of positions where both A and B were
    recognised for the same move; the diagonal = positions with tactic A at all."""
    sets = [sorted(set(_split(s))) for s in df[col]]
    names = sorted({t for s in sets for t in s})
    m = pd.DataFrame(0, index=names, columns=names, dtype=int)
    for s in sets:
        for a, b in itertools.product(s, s):
            m.loc[a, b] += 1
    return m


def plot_cooccurrence(df: pd.DataFrame, col: str = "missed_tactics", normalise: bool = True):
    """Heatmap of co-occurrence. normalise=True: cell = P(column tactic | row tactic)."""
    m = cooccurrence_matrix(df, col)
    fig, ax = plt.subplots(figsize=(0.9 * len(m) + 3, 0.8 * len(m) + 2))
    if m.empty:
        ax.text(0.5, 0.5, "no labelled positions", ha="center")
        return fig
    val = m.div(np.diag(m.to_numpy()), axis=0) if normalise else m
    im = ax.imshow(val.to_numpy(dtype=float), cmap="Blues", vmin=0, vmax=1 if normalise else None)
    for i in range(len(m)):
        for j in range(len(m)):
            ax.text(j, i, f"{m.iloc[i, j]}" + (f"\n{val.iloc[i, j]:.0%}" if normalise else ""),
                    ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(m)))
    ax.set_xticklabels(m.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(m)))
    ax.set_yticklabels(m.index, fontsize=8)
    ax.set_title(f"Co-occurrence of tactics in one move ({col})"
                 + ("\ncell = share of the ROW tactic's positions that also have the COLUMN tactic" if normalise else ""))
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. tactic_length
# ---------------------------------------------------------------------------

def _length_group(s: pd.Series, cap: int = 3) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    return x.map(lambda v: np.nan if pd.isna(v) else (f"{cap}+" if v >= cap else str(int(v))))


def length_distribution(df: pd.DataFrame, by: str = None, cap: int = 3) -> pd.DataFrame:
    """Share of positions per tactic length (0, 1, 2, 3+), optionally per `by`
    column (e.g. 'missed_primary' or 'elo_bin'), with 95% CI."""
    d = df.copy()
    d["len_group"] = _length_group(d["tactic_length"], cap)
    d = d.dropna(subset=["len_group"])
    keys = ([by] if by else [])
    g = d.groupby(keys + ["len_group"], observed=True).size().rename("n").reset_index()
    tot = g.groupby(keys, observed=True)["n"].transform("sum") if keys else g["n"].sum()
    g["total"] = tot
    g["share"] = g["n"] / g["total"]
    return _with_ci(g, "n", "total")


def blunder_by_length(df: pd.DataFrame, by: list = None, cap: int = 3) -> pd.DataFrame:
    """n_critical, n_found, blunder_rate (+95% CI) per tactic length (and `by` columns)."""
    d = df.copy()
    d["len_group"] = _length_group(d["tactic_length"], cap)
    d = d.dropna(subset=["len_group"])
    keys = ["len_group"] + (by or [])
    g = d.groupby(keys, observed=True).agg(n_critical=("found_best", "size"),
                                            n_found=("found_best", "sum")).reset_index()
    return add_blunder_columns(g)


def blunder_by_tactic_and_length(df: pd.DataFrame, col: str = "missed_tactics", cap: int = 3) -> pd.DataFrame:
    """Blunder rate for every (tactic, length) pair (multi-label)."""
    d = df.copy()
    d["len_group"] = _length_group(d["tactic_length"], cap)
    d = d.dropna(subset=["len_group"])
    long = explode_labels(d, col)
    g = long.groupby(["tactic", "len_group"], observed=True).agg(
        n_critical=("found_best", "size"), n_found=("found_best", "sum")).reset_index()
    return add_blunder_columns(g)


def plot_blunder_by_length(df: pd.DataFrame, cap: int = 3):
    t = blunder_by_length(df, cap=cap).sort_values("len_group")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(t))
    r = t["blunder_rate"].to_numpy(dtype=float)
    ax.bar(x, r, yerr=_err(r, t["blunder_ci_low"], t["blunder_ci_high"]), capsize=4, color="#c0504d")
    for xi, v, n in zip(x, r, t["n_critical"]):
        ax.text(xi, min(v + 0.03, 1.0), f"{v:.0%}\nn={int(n)}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(t["len_group"])
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("tactic length (consecutive critical positions for the player)")
    ax.set_ylabel("blunder rate")
    ax.set_title("Blunder rate by tactic length (95% CI)")
    fig.tight_layout()
    return fig


def plot_length_distribution(df: pd.DataFrame, by: str = "missed_primary", cap: int = 3, min_n: int = 20):
    """Stacked bars: distribution of tactic length for every primary tactic."""
    g = length_distribution(df, by=by, cap=cap)
    tot = g.groupby(by)["n"].sum()
    keep = tot[tot >= min_n].sort_values(ascending=False).index
    p = g[g[by].isin(keep)].pivot(index=by, columns="len_group", values="share").reindex(keep).fillna(0)
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(p) + 2))
    left = np.zeros(len(p))
    for c in sorted(p.columns):
        ax.barh(np.arange(len(p)), p[c], left=left, label=f"length {c}")
        left += p[c].to_numpy()
    ax.set_yticks(np.arange(len(p)))
    ax.set_yticklabels([f"{i} (n={int(tot[i])})" for i in p.index], fontsize=8)
    ax.set_xlabel("share of positions")
    ax.set_title("Tactic length by primary tactic")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Unforced errors: tactics
# ---------------------------------------------------------------------------

def load_unforced(unforced_csv: str) -> pd.DataFrame:
    u = pd.read_csv(unforced_csv)
    u["is_unforced_error"] = _truthy(u["is_unforced_error"])
    for c in ("missed_tactics", "allowed_tactics", "missed_primary", "allowed_primary"):
        if c in u.columns:
            u[c] = u[c].fillna("")
    return u


def tactic_share_overall(events: pd.DataFrame, col: str) -> pd.DataFrame:
    """Share of `events` carrying each tactic (multi-label; 'none' = no label), with 95% CI."""
    long = explode_labels(events, col)
    g = long.groupby("tactic").size().rename("n").reset_index()
    g["total"] = len(events)
    g["share"] = g["n"] / g["total"]
    return _with_ci(g, "n", "total").sort_values("n", ascending=False).reset_index(drop=True)


def compare_error_types(critical: pd.DataFrame, unforced: pd.DataFrame) -> pd.DataFrame:
    """Side by side: tactics ALLOWED by mistakes in critical positions vs. by unforced
    errors, and tactics MISSED in both. Columns: <kind>_<crit|unf>_{n,share,ci}."""
    mistakes = critical[~critical["found_best"]]
    errors = unforced[unforced["is_unforced_error"]]
    parts = []
    for kind, col in (("allowed", "allowed_tactics"), ("missed", "missed_tactics")):
        for tag, ev in (("crit", mistakes), ("unf", errors)):
            t = tactic_share_overall(ev, col).set_index("tactic")
            t.columns = [f"{kind}_{tag}_{c}" for c in t.columns]
            parts.append(t[[f"{kind}_{tag}_n", f"{kind}_{tag}_share", f"{kind}_{tag}_ci_low", f"{kind}_{tag}_ci_high"]])
    return pd.concat(parts, axis=1).fillna(0).sort_index()


def plot_compare_error_types(critical: pd.DataFrame, unforced: pd.DataFrame, kind: str = "allowed"):
    """Grouped bars: share of mistakes (critical) vs unforced errors with the tactic."""
    t = compare_error_types(critical, unforced)
    t = t.sort_values(f"{kind}_crit_share", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(t))
    for k, (tag, name, col) in enumerate((("crit", "mistakes in critical positions", "#c0504d"),
                                          ("unf", "unforced errors", "#8e44ad"))):
        r = t[f"{kind}_{tag}_share"].to_numpy(dtype=float)
        lo, hi = t[f"{kind}_{tag}_ci_low"], t[f"{kind}_{tag}_ci_high"]
        ax.bar(x + (k - 0.5) * 0.4, r, 0.4, yerr=_err(r, lo, hi), capsize=2, label=name, color=col)
    ax.set_xticks(x)
    ax.set_xticklabels(t.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("share of errors")
    ax.set_title(f"Tactics {kind} - critical-position mistakes vs unforced errors (95% CI)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Coverage of the classification
# ---------------------------------------------------------------------------

def coverage_summary(critical: pd.DataFrame, unforced: Optional[pd.DataFrame] = None,
                     target: float = 0.80) -> pd.DataFrame:
    """Share of positions with >= 1 recognised tactic:
       - all critical positions (missed label of the best move)
       - blunders in critical positions (missed OR allowed label)
       - unforced errors (missed OR allowed)
    with 95% CI and whether the supervisor's target is reached (CI low >= target)."""
    rows = []

    def add(name, mask_total, mask_hit):
        n, k = int(mask_total.sum()), int((mask_total & mask_hit).sum())
        lo, hi = wilson(k, n) if n else (np.nan, np.nan)
        rows.append({"set": name, "n": n, "classified": k, "share": k / n if n else np.nan,
                     "ci_low": lo, "ci_high": hi, f"target_{target:.0%}": (k / n >= target) if n else None})

    has_m = critical["missed_tactics"].fillna("").ne("")
    has_a = critical["allowed_tactics"].fillna("").ne("")
    allc = pd.Series(True, index=critical.index)
    add("all critical positions (missed label)", allc, has_m)
    add("blunders (missed or allowed)", ~critical["found_best"], has_m | has_a)
    add("blunders (missed only)", ~critical["found_best"], has_m)
    add("blunders (allowed only)", ~critical["found_best"], has_a)
    if unforced is not None:
        e = unforced["is_unforced_error"]
        um = unforced["missed_tactics"].fillna("").ne("")
        ua = unforced["allowed_tactics"].fillna("").ne("")
        add("unforced errors (missed or allowed)", e, um | ua)
    return pd.DataFrame(rows)


def primary_breakdown(critical: pd.DataFrame, col: str = "missed_primary") -> pd.DataFrame:
    """Which primary tactic labels the positions got ('none' = unexplained)."""
    d = critical[col].replace("", "none").value_counts().rename("n").reset_index()
    d.columns = ["primary", "n"]
    d["total"] = len(critical)
    d["share"] = d["n"] / d["total"]
    return _with_ci(d, "n", "total")


# ---------------------------------------------------------------------------
# 5. Full report
# ---------------------------------------------------------------------------

def full_report(main_csv: str, enriched_csv: str, unforced_csv: Optional[str], registry_csv: Optional[str],
                bins: list, labels: list, out_dir: str = "report", show: bool = True,
                min_n: int = 10) -> dict:
    """Builds every table and plot; saves CSVs and PNGs into out_dir; returns
    {'data': critical df, 'unforced': df or None, 'tables': {...}, 'figures': {...}}."""
    os.makedirs(out_dir, exist_ok=True)
    df = load_tactic_data(main_csv, enriched_csv, registry_csv)
    unf = load_unforced(unforced_csv) if unforced_csv and os.path.exists(unforced_csv) else None
    mistakes = df[~df["found_best"]]

    tables, figs = {}, {}

    # blunder rate per tactic (missed) and per tactic x Elo
    tables["blunder_by_tactic"] = blunder_by_tactic(df, "missed_tactics")
    tables["blunder_by_tactic_elo"] = blunder_by_tactic(with_elo_bins(df, bins, labels), "missed_tactics", ["elo_bin"])
    figs["blunder_by_tactic"] = plot_tactic_blunder_rates(df, "missed_tactics", min_n)
    figs["blunder_by_tactic_elo"] = plot_tactic_elo_heatmap(df, bins, labels, "missed_tactics", min_n)

    # allowed tactics
    tables["allowed_share"] = share_table(mistakes, bins, labels, "allowed_tactics")
    figs["allowed_share"] = plot_share_heatmap(mistakes, bins, labels, "allowed_tactics", min_n=min_n)

    # 1. co-occurrence
    for col in ("missed_tactics", "allowed_tactics"):
        src = df if col == "missed_tactics" else mistakes
        tables[f"combinations_{col}"] = combination_table(src, col)
        tables[f"cooccurrence_{col}"] = cooccurrence_matrix(src, col)
        figs[f"cooccurrence_{col}"] = plot_cooccurrence(src, col)

    # 2. tactic length
    if "tactic_length" in df.columns and df["tactic_length"].notna().any():
        tables["length_distribution"] = length_distribution(df)
        tables["length_by_primary"] = length_distribution(df, by="missed_primary")
        tables["blunder_by_length"] = blunder_by_length(df)
        tables["blunder_by_length_elo"] = blunder_by_length(with_elo_bins(df, bins, labels), by=["elo_bin"])
        tables["blunder_by_tactic_length"] = blunder_by_tactic_and_length(df)
        figs["blunder_by_length"] = plot_blunder_by_length(df)
        figs["length_by_primary"] = plot_length_distribution(df)
    else:
        print("no tactic_length in the data - run enrich_critical(..., compute_length=True)")

    # 4. coverage
    tables["coverage"] = coverage_summary(df, unf)
    tables["primary_missed"] = primary_breakdown(df, "missed_primary")
    tables["classified_by_elo"] = classified_table(mistakes, bins, labels).reset_index()
    figs["classified_share"] = plot_classified_share(mistakes, bins, labels)

    # 3. unforced errors
    if unf is not None:
        tables["unforced_rate"] = unforced_rate_table(unf, bins, labels).reset_index()
        figs["unforced_rate"] = plot_unforced_rate(unf, bins, labels)
        errs = unf[unf["is_unforced_error"]]
        if len(errs):
            tables["unforced_allowed_share"] = share_table(errs, bins, labels, "allowed_tactics")
            tables["unforced_missed_share"] = share_table(errs, bins, labels, "missed_tactics")
            tables["compare_error_types"] = compare_error_types(df, unf).reset_index()
            figs["compare_allowed"] = plot_compare_error_types(df, unf, "allowed")
            figs["compare_missed"] = plot_compare_error_types(df, unf, "missed")

    for name, t in tables.items():
        t.to_csv(os.path.join(out_dir, f"{name}.csv"), index=name.startswith("cooccurrence"))
    for name, f in figs.items():
        f.savefig(os.path.join(out_dir, f"{name}.png"), dpi=150, bbox_inches="tight")
        if not show:
            plt.close(f)

    print(f"saved {len(tables)} tables and {len(figs)} figures to {out_dir}/")
    print(tables["coverage"].round(3).to_string(index=False))
    return {"data": df, "unforced": unf, "tables": tables, "figures": figs}
