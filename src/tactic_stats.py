"""
tactic_stats.py

Tables and plots built on the output of enrich.py. Everything shows a
95% Wilson confidence interval.

Two views of a critical position:
  missed   - the tactics realised by the BEST move.  For every tactic T:
             blunder rate = share of positions (where the best move realises T)
             in which the player did NOT play the best move.
  allowed  - after a MISTAKE, the tactics the opponent gets. For every T:
             share of the mistakes that allowed T.

A position can match several tactics; in the tables it counts once for each
(multi-label). Positions with no recognised tactic are labelled "none" -
their share is the part of the data the rules do not explain yet.
"""

from __future__ import annotations

import chess
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from buckets import CMAP, _annotated_heatmap, _err, add_blunder_columns, wilson
from registry import load_registry


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _truthy(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().eq("true")


def load_tactic_data(main_csv: str, enriched_csv: str, registry_csv: str = None) -> pd.DataFrame:
    """Enriched critical positions joined with the main CSV (fen, moves,
    pieces, scores) and - if a registry is given - a Lichess link to the
    game (`lichess_link`, opens at the position before the move)."""
    main = pd.read_csv(main_csv)
    main["is_critical"] = _truthy(main["is_critical"])
    cols = ["game_id", "ply", "fen", "best_move", "played_move", "best_piece", "played_piece", "best_cp", "second_cp"]
    enr = pd.read_csv(enriched_csv)
    enr["found_best"] = _truthy(enr["found_best"])
    df = enr.merge(main[main["is_critical"]][cols], on=["game_id", "ply"], how="left")
    for c in ("missed_tactics", "allowed_tactics", "missed_primary", "allowed_primary", "opp_pv", "opp_reply", "pv_best"):
        if c in df.columns:
            df[c] = df[c].fillna("")
    if registry_csv:
        reg = load_registry(registry_csv).drop_duplicates("game_id").set_index("game_id")["game_key"]
        key = df["game_id"].map(reg)
        df["lichess_link"] = [f"{k}#{p - 1}" if isinstance(k, str) and k.startswith("http") else ""
                              for k, p in zip(key, df["ply"])]
    return df


def with_elo_bins(df: pd.DataFrame, bins: list, labels: list) -> pd.DataFrame:
    d = df.copy()
    d["elo_bin"] = pd.cut(d["player_elo"], bins=bins, labels=labels)
    return d


def explode_labels(df: pd.DataFrame, col: str = "missed_tactics") -> pd.DataFrame:
    """One row per (position, tactic); positions without a tactic get 'none'."""
    d = df.copy()
    d["tactic"] = d[col].fillna("").map(lambda s: s.split(";") if s else ["none"])
    return d.explode("tactic")


# ---------------------------------------------------------------------------
# Missed tactics: blunder rate per tactic (and Elo)
# ---------------------------------------------------------------------------

def blunder_by_tactic(df: pd.DataFrame, col: str = "missed_tactics", by: list = None) -> pd.DataFrame:
    """n_critical, n_found, blunder_rate (+95% CI) per tactic (and `by` columns)."""
    long = explode_labels(df, col)
    keys = ["tactic"] + (by or [])
    g = long.groupby(keys, observed=True).agg(n_critical=("found_best", "size"), n_found=("found_best", "sum")).reset_index()
    return add_blunder_columns(g)


def plot_tactic_blunder_rates(df: pd.DataFrame, col: str = "missed_tactics", min_n: int = 10):
    """Horizontal bars: blunder rate (95% CI) for every tactic."""
    t = blunder_by_tactic(df, col)
    t = t[t["n_critical"] >= min_n].sort_values("blunder_rate")
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(t) + 1.5))
    y = np.arange(len(t))
    r = t["blunder_rate"].to_numpy(dtype=float)
    xerr = _err(r, t["blunder_ci_low"], t["blunder_ci_high"])
    ax.barh(y, r, xerr=xerr, capsize=3, color=["#7f8c8d" if x == "none" else "#c0504d" for x in t["tactic"]])
    for yi, v, n in zip(y, r, t["n_critical"]):
        ax.text(min(v + 0.02, 0.97), yi, f"{v:.0%}  n={n}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(t["tactic"])
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("blunder rate (share of critical positions where the best move was NOT played)")
    ax.set_title("Blunder rate by tactic realised by the best move (95% CI; 'none' = no tactic recognised)")
    fig.tight_layout()
    return fig


def plot_tactic_elo_heatmap(df: pd.DataFrame, bins: list, labels: list, col: str = "missed_tactics", min_n: int = 10):
    """Heatmap tactic x Elo bin: blunder rate, [95% CI], n."""
    t = blunder_by_tactic(with_elo_bins(df, bins, labels), col, ["elo_bin"])
    order = (t.groupby("tactic")["n_critical"].sum().sort_values(ascending=False).index.tolist())
    def piv(v):
        return t.pivot(index="tactic", columns="elo_bin", values=v).reindex(index=order, columns=labels)
    fig, ax = plt.subplots(figsize=(1.6 * len(labels) + 3, 0.7 * len(order) + 1.5))
    _annotated_heatmap(ax, piv("blunder_rate"), piv("n_critical"), min_n,
                       f"Blunder rate by tactic and Elo (grey: n < {min_n})", show_cbar=True,
                       lo=piv("blunder_ci_low"), hi=piv("blunder_ci_high"), fontsize=7)
    fig.tight_layout()
    return fig


def plot_tactic_elo_lines(df: pd.DataFrame, bins: list, labels: list, col: str = "missed_tactics",
                          min_n: int = 15, top: int = 6):
    """Blunder rate vs Elo, one line per tactic (the `top` most frequent
    ones; bins with fewer than min_n cases are left out), 95% CI bars."""
    t = blunder_by_tactic(with_elo_bins(df, bins, labels), col, ["elo_bin"])
    t = t[t["n_critical"] >= min_n]
    tactics = (t.groupby("tactic")["n_critical"].sum().sort_values(ascending=False).head(top).index)
    fig, ax = plt.subplots(figsize=(9, 5))
    for k, name in enumerate(tactics):
        sub = t[t["tactic"] == name].set_index("elo_bin").reindex(labels)
        xs = np.arange(len(labels)) + (k - len(tactics) / 2) * 0.04
        r = sub["blunder_rate"].to_numpy(dtype=float)
        ax.errorbar(xs, r, yerr=_err(r, sub["blunder_ci_low"], sub["blunder_ci_high"]), marker="o", capsize=3, label=name)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("blunder rate")
    ax.set_title("Blunder rate vs Elo, by tactic (95% CI)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Allowed tactics / classified share
# ---------------------------------------------------------------------------

def share_table(events: pd.DataFrame, bins: list, labels: list, col: str = "allowed_tactics") -> pd.DataFrame:
    """For a table of 'events' (e.g. all mistakes): per Elo bin and tactic,
    the share of events carrying that tactic, with a 95% Wilson interval."""
    e = with_elo_bins(events, bins, labels)
    totals = e.groupby("elo_bin", observed=True).size()
    long = explode_labels(e, col)
    cnt = long.groupby(["tactic", "elo_bin"], observed=True).size().rename("n").reset_index()
    cnt["total"] = cnt["elo_bin"].map(totals).astype(float)
    cnt["share"] = cnt["n"] / cnt["total"]
    ci = [wilson(k, n) for k, n in zip(cnt["n"], cnt["total"])]
    cnt["ci_low"] = [c[0] for c in ci]
    cnt["ci_high"] = [c[1] for c in ci]
    return cnt


def plot_share_heatmap(events: pd.DataFrame, bins: list, labels: list, col: str = "allowed_tactics",
                       title: str = "Share of mistakes that allow the tactic", min_n: int = 10):
    """Heatmap tactic x Elo: share of events with the tactic, [95% CI], total."""
    t = share_table(events, bins, labels, col)
    order = t.groupby("tactic")["n"].sum().sort_values(ascending=False).index.tolist()
    def piv(v):
        return t.pivot(index="tactic", columns="elo_bin", values=v).reindex(index=order, columns=labels)
    vmax = max(0.05, float(np.nanmax(piv("share").to_numpy(dtype=float))))
    fig, ax = plt.subplots(figsize=(1.6 * len(labels) + 3, 0.7 * len(order) + 1.5))
    _annotated_heatmap(ax, piv("share"), piv("total"), min_n, f"{title} (grey: < {min_n} events)",
                       show_cbar=True, lo=piv("ci_low"), hi=piv("ci_high"), fontsize=7,
                       vmax=vmax, cmap="Blues", cbar_label="share")
    fig.tight_layout()
    return fig


def classified_table(mistakes: pd.DataFrame, bins: list, labels: list) -> pd.DataFrame:
    """For mistakes: per Elo bin, how many have a recognised tactic - missed,
    allowed, both, or none - with the share classified (>= 1 label) and its CI."""
    m = with_elo_bins(mistakes, bins, labels)
    has_missed = m["missed_tactics"].fillna("").ne("")
    has_allowed = m["allowed_tactics"].fillna("").ne("")
    m["class"] = np.select([has_missed & has_allowed, has_missed, has_allowed],
                           ["both", "missed only", "allowed only"], default="none")
    tab = pd.crosstab(m["elo_bin"], m["class"]).reindex(columns=["both", "missed only", "allowed only", "none"], fill_value=0)
    tab["total"] = tab.sum(axis=1)
    tab["classified"] = tab["total"] - tab["none"]
    tab["classified_share"] = tab["classified"] / tab["total"].where(tab["total"] > 0)
    ci = [wilson(k, n) for k, n in zip(tab["classified"], tab["total"])]
    tab["ci_low"] = [c[0] for c in ci]
    tab["ci_high"] = [c[1] for c in ci]
    return tab


def plot_classified_share(mistakes: pd.DataFrame, bins: list, labels: list):
    """Stacked bars per Elo bin: how the mistakes split into both / missed only /
    allowed only / none, with the classified share and its 95% CI on top."""
    tab = classified_table(mistakes, bins, labels).reindex(labels)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(tab))
    bottom = np.zeros(len(tab))
    colors = {"both": "#2e7d32", "missed only": "#66bb6a", "allowed only": "#a5d6a7", "none": "#bdbdbd"}
    for c in ["both", "missed only", "allowed only", "none"]:
        share = (tab[c] / tab["total"]).to_numpy(dtype=float)
        ax.bar(x, np.nan_to_num(share), bottom=bottom, color=colors[c], label=c)
        bottom += np.nan_to_num(share)
    for i, (s, lo, hi, n) in enumerate(zip(tab["classified_share"], tab["ci_low"], tab["ci_high"], tab["total"])):
        if not np.isnan(s):
            ax.text(i, 1.02, f"{s:.0%} classified\n[{lo:.0%}-{hi:.0%}]  n={int(n)}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.2)
    ax.set_title("Mistakes: share with a recognised tactic (missed = best move, allowed = opponent's reply)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Owner and length of a tactic
# ---------------------------------------------------------------------------

def add_owner(df: pd.DataFrame) -> pd.DataFrame:
    """Column tactic_owner: 'player' (best move realises a tactic = missed),
    'opponent' (the played move gives one away = allowed), 'both' or 'none'.
    Computed from the label columns, so it also works on older result files."""
    d = df.copy()
    m = d["missed_tactics"].fillna("").ne("")
    a = d["allowed_tactics"].fillna("").ne("")
    d["tactic_owner"] = np.select([m & a, m, a], ["both", "player", "opponent"], default="none")
    return d


def length_table(df: pd.DataFrame, bins: list, labels: list, by: str = "elo") -> pd.DataFrame:
    """Distribution of tactic_length (1 = single only-move, 2 = a preparatory
    forcing move first, ...). by='elo' -> per Elo bin, by='primary' -> per
    primary missed tactic. Counts and shares per length."""
    e = with_elo_bins(df, bins, labels)
    e = e[e["tactic_length"].notna()]
    key = "elo_bin" if by == "elo" else "missed_primary"
    tab = pd.crosstab(e[key], e["tactic_length"].astype(int))
    tab["total"] = tab.sum(axis=1)
    share = tab.drop(columns="total").div(tab["total"], axis=0).add_prefix("share_")
    return pd.concat([tab, share], axis=1)


def plot_length_distribution(df: pd.DataFrame, bins: list, labels: list, by: str = "elo"):
    """Stacked bars: share of each tactic length per Elo bin (or per tactic)."""
    tab = length_table(df, bins, labels, by)
    cols = [c for c in tab.columns if str(c).startswith("share_")]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bottom = np.zeros(len(tab))
    for c in cols:
        v = tab[c].to_numpy(dtype=float)
        ax.bar(range(len(tab)), v, bottom=bottom, label=f"length {c[6:]}")
        bottom += v
    ax.set_xticks(range(len(tab)))
    ax.set_xticklabels([f"{i}\nn={int(n)}" for i, n in zip(tab.index, tab["total"])], rotation=0 if by == "elo" else 30, ha="center" if by == "elo" else "right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("share")
    ax.set_title("Tactic length (consecutive critical positions)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Unforced errors
# ---------------------------------------------------------------------------

def unforced_rate_table(df: pd.DataFrame, bins: list, labels: list) -> pd.DataFrame:
    """Unforced-error rate per Elo bin = errors / non-critical positions
    (rows of the output of find_unforced_errors), with 95% CI."""
    d = with_elo_bins(df, bins, labels)
    d["is_unforced_error"] = _truthy(d["is_unforced_error"])
    g = d.groupby("elo_bin", observed=True).agg(positions=("is_unforced_error", "size"),
                                                 errors=("is_unforced_error", "sum")).reindex(labels)
    g["rate"] = g["errors"] / g["positions"].where(g["positions"] > 0)
    ci = [wilson(k, n) for k, n in zip(g["errors"].fillna(0), g["positions"].fillna(0))]
    g["ci_low"] = [c[0] for c in ci]
    g["ci_high"] = [c[1] for c in ci]
    return g


def plot_unforced_rate(df: pd.DataFrame, bins: list, labels: list):
    g = unforced_rate_table(df, bins, labels)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(g))
    r = g["rate"].to_numpy(dtype=float)
    ax.bar(x, np.nan_to_num(r), yerr=_err(r, g["ci_low"], g["ci_high"]), capsize=4, color="#8e44ad")
    for i, (v, n, k) in enumerate(zip(r, g["positions"], g["errors"])):
        if not np.isnan(v):
            ax.text(i, v + 0.004, f"{v:.1%}\n{int(k)}/{int(n)}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, float(np.nanmax(r)) * 1.5 if np.isfinite(r).any() else 1)
    ax.set_ylabel("unforced errors / non-critical positions")
    ax.set_title("Unforced-error rate by Elo (95% CI)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Review samples (for checking the classification by hand)
# ---------------------------------------------------------------------------

def review_sample(df: pd.DataFrame, tactic: str = None, kind: str = "missed", n: int = 20, seed: int = 0,
                  mistakes_only: bool = True) -> pd.DataFrame:
    """Random sample of classified positions. kind='missed' looks at
    missed_tactics (the best move), kind='allowed' at allowed_tactics.
    tactic=None -> any classified position."""
    col = f"{kind}_tactics"
    d = df[~df["found_best"]] if mistakes_only else df
    has = d[col].fillna("").ne("")
    if tactic is not None:
        has = d[col].fillna("").map(lambda s: tactic in s.split(";"))
    d = d[has]
    return d.sample(n=min(n, len(d)), random_state=seed)


def show_review(sample: pd.DataFrame, kind: str = "missed", size: int = 280) -> None:
    """Draw every row of `sample` with the detected tactics (needs tactics_inspect).
    kind='missed':  the board with the best move.
    kind='allowed': TWO boards side by side - 1) the mistake (blue = best move, red = the move
                    played), 2) the position after it with the opponent's reply
                    and the tactic that was detected."""
    from IPython.display import HTML, display
    import chess.svg
    from tactics_inspect import BLUE, RED, _cell, inspect_move
    for r in sample.itertuples():
        link = getattr(r, "lichess_link", "") or ""
        head = f"game_id {r.game_id}, ply {r.ply}, Elo {r.player_elo}  {link}"
        display(HTML(f"<div style='margin-top:14px;font-family:monospace'><b>{head}</b></div>"))
        if kind == "missed":
            inspect_move(r.fen, r.best_move, size=size, pv=r.pv_best.split() if r.pv_best else None,
                         best_cp=r.best_cp, second_cp=r.second_cp)
            continue
        b0 = chess.Board(r.fen)
        best, played = chess.Move.from_uci(r.best_move), chess.Move.from_uci(r.played_move)
        cap = (f"<b>1) the mistake</b>: played {b0.san(played)} (red), best was {b0.san(best)} (blue)")
        arrows = [chess.svg.Arrow(best.from_square, best.to_square, color=BLUE),
                  chess.svg.Arrow(played.from_square, played.to_square, color=RED)]
        cells = [_cell(b0, arrows, cap, "#c02020", size)]
        b = b0.copy(stack=False)
        b.push(played)
        if r.opp_reply:
            _, cell2 = inspect_move(b.fen(), r.opp_reply, size=size, html_only=True,
                                    pv=r.opp_pv.split() if r.opp_pv else None,
                                    best_cp=None if pd.isna(r.opp_cp) else r.opp_cp)
            cells.append(cell2)
        display(HTML("<div style='display:flex;flex-wrap:wrap;align-items:flex-start'>" + "".join(cells) + "</div>"))
