"""
registry.py

A registry of games that were already analysed with Stockfish.

Why: we run several different analyses over the same PGN file. The registry
(one CSV file, `analysed_games.csv`) remembers every analysed game, so the
next analysis can skip it, and it records how many critical positions the
game had:

    complete = True   -> every eligible position of the game was analysed,
                         `n_critical` is the exact number (0, 1, 2, ...)
    complete = False  -> only a part of the game was analysed (e.g. we
                         stopped after finding 2 critical positions, or
                         skipped positions whose bucket was already full).
                         `n_critical` is only "at least this many" -> the
                         true number is unknown ("nie wiadomo").

A game is identified by its Lichess URL (the `Site` header), which is
stable; if there is none, by "<file name>#<index in the file>".

The CSV is the working file (fast to append to). `export_registry_xlsx`
writes the same data as an Excel workbook with separate sheets.
"""

from __future__ import annotations

import io
import os
from datetime import datetime
from typing import Iterator, Optional

import pandas as pd

REG_COLUMNS = [
    "game_key", "source_file", "game_id", "white_elo", "black_elo",
    "analysed_positions", "eligible_positions", "n_critical", "complete",
    "method", "depth", "gap_threshold", "min_ply", "analysed_at",
]


def make_game_key(headers, source_file: str, game_id: int) -> str:
    """Stable identifier of a game (Lichess URL, else file name + index)."""
    site = (headers.get("Site", "") or "")
    if site.startswith("http"):
        return site
    return f"{os.path.basename(source_file)}#{game_id}"


def make_row(
    game_key: str,
    source_file: str,
    game_id: int,
    white_elo,
    black_elo,
    analysed_positions,
    eligible_positions,
    n_critical,
    complete: bool,
    method: str,
    depth,
    gap_threshold,
    min_ply,
) -> dict:
    return {
        "game_key": game_key,
        "source_file": os.path.basename(source_file),
        "game_id": game_id,
        "white_elo": white_elo,
        "black_elo": black_elo,
        "analysed_positions": analysed_positions,
        "eligible_positions": eligible_positions,
        "n_critical": n_critical,
        "complete": bool(complete),
        "method": method,
        "depth": depth,
        "gap_threshold": gap_threshold,
        "min_ply": min_ply,
        "analysed_at": datetime.now().isoformat(timespec="seconds"),
    }


def append_registry(path: str, rows: list[dict]) -> None:
    """Append rows to the registry CSV (creates it with a header if new)."""
    if not rows:
        return
    df = pd.DataFrame(rows).reindex(columns=REG_COLUMNS)
    if os.path.exists(path):
        df.to_csv(path, mode="a", index=False, header=False)
    else:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        df.to_csv(path, index=False)


def load_registry(path: str) -> pd.DataFrame:
    """Read the registry; one row per game (best entry wins: complete first,
    then the larger number of critical positions found)."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=REG_COLUMNS)
    reg = pd.read_csv(path)
    reg["complete"] = reg["complete"].astype(str).str.lower().eq("true")
    reg = reg.sort_values(["complete", "n_critical"], ascending=[False, False], kind="stable")
    return reg.drop_duplicates("game_key", keep="first").reset_index(drop=True)


def skip_keys(path: Optional[str], mode: str = "all") -> Optional[set]:
    """Keys of games to skip. mode='all' skips every registered game,
    mode='complete' only games that were analysed completely."""
    if not path:
        return None
    reg = load_registry(path)
    if mode == "complete":
        reg = reg[reg["complete"]]
    return set(reg["game_key"])


def critical_distribution(reg: pd.DataFrame, max_exact: int = 5) -> pd.DataFrame:
    """How many games have 0, 1, 2, ... critical positions (certain), and
    how many are 'unknown' because the analysis of the game was partial."""
    rows = []
    comp = reg[reg["complete"]]
    for k in range(max_exact):
        rows.append((f"{k} (pewne)", int((comp["n_critical"] == k).sum())))
    rows.append((f"{max_exact}+ (pewne)", int((comp["n_critical"] >= max_exact).sum())))
    inc = reg[~reg["complete"]]
    rows.append(("nie wiadomo - analiza niepelna, znaleziono 0", int((inc["n_critical"] == 0).sum())))
    rows.append(("nie wiadomo - analiza niepelna, znaleziono 1 lub wiecej", int((inc["n_critical"] >= 1).sum())))
    out = pd.DataFrame(rows, columns=["liczba_pozycji_krytycznych", "partie"])
    total = out["partie"].sum()
    out["udzial"] = (out["partie"] / total).round(4) if total else 0.0
    return out


def _iter_headers(path: str) -> Iterator:
    """Iterate over the headers of every game in a .pgn / .pgn.zst file,
    in the same order as iter_games() in critical.py (fast, no moves)."""
    import chess.pgn

    def _loop(handle):
        while True:
            headers = chess.pgn.read_headers(handle)
            if headers is None:
                break
            yield headers

    if path.endswith(".zst"):
        import zstandard as zstd

        with open(path, "rb") as fh:
            with zstd.ZstdDecompressor().stream_reader(fh) as reader:
                text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
                yield from _loop(text)
    else:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            yield from _loop(f)


def _per_game_from_csv(path: str, kind: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame()
    df["is_critical"] = df["is_critical"].astype(str).str.lower().eq("true")
    agg = {"n_critical": ("is_critical", "sum")}
    if kind == "complete":
        agg["analysed_positions"] = ("is_critical", "size")
    for col in ("white_elo", "black_elo"):
        if col in df.columns:
            agg[col] = (col, "first")
    g = df.groupby("game_id").agg(**agg).reset_index()
    if kind != "complete":
        g["analysed_positions"] = float("nan")
    g["complete"] = kind == "complete"
    for col in ("white_elo", "black_elo"):
        if col not in g.columns:
            g[col] = float("nan")
    return g


def backfill_registry(
    pgn_path: str,
    registry_csv: str,
    sources: dict,
    depth: int = 12,
    gap_threshold: float = 0.20,
    min_ply: int = 8,
) -> int:
    """
    One-off: put games analysed BEFORE the registry existed into it.

    `sources` maps a results CSV to its kind:
        'complete' - the CSV has every analysed position of each game
                     (made by find_critical_positions) -> exact counts
        'partial'  - the CSV has only some / only critical positions
                     (collect_critical_* functions) -> counts are 'at least'

    Games are matched through their index in `pgn_path`, so `pgn_path` MUST
    be the same file that was analysed. Returns the number of rows added.
    """
    per_kind = {}
    for csv_path, kind in sources.items():
        if not os.path.exists(csv_path):
            print("pomijam (brak pliku):", csv_path)
            continue
        g = _per_game_from_csv(csv_path, kind)
        if not g.empty:
            per_kind[csv_path] = g
    if not per_kind:
        print("nic do dopisania")
        return 0

    all_ids = set()
    for g in per_kind.values():
        all_ids |= set(g["game_id"].astype(int))
    max_id = max(all_ids)

    keys = {}
    for idx, headers in enumerate(_iter_headers(pgn_path)):
        if idx in all_ids:
            keys[idx] = make_game_key(headers, pgn_path, idx)
        if idx >= max_id:
            break

    existing = load_registry(registry_csv).set_index("game_key")["complete"].to_dict()
    new_rows = {}
    for csv_path, g in per_kind.items():
        for r in g.itertuples(index=False):
            gid = int(r.game_id)
            key = keys.get(gid)
            if key is None:
                continue
            complete = bool(r.complete)
            if key in existing and (existing[key] or not complete):
                continue                      # registry already has an equal/better entry
            if key in new_rows and (new_rows[key]["complete"] or not complete):
                continue
            n_found = int(r.n_critical)
            analysed = None if pd.isna(r.analysed_positions) else int(r.analysed_positions)
            new_rows[key] = make_row(
                key, pgn_path, gid,
                None if pd.isna(r.white_elo) else int(r.white_elo),
                None if pd.isna(r.black_elo) else int(r.black_elo),
                analysed, analysed if complete else None,
                n_found, complete, "backfill", depth, gap_threshold, min_ply,
            )
    append_registry(registry_csv, list(new_rows.values()))
    print(f"dopisano {len(new_rows)} partii do rejestru")
    return len(new_rows)


def export_registry_xlsx(registry_csv: str, xlsx_path: str) -> None:
    """Write the registry as an Excel workbook with separate sheets:
    'partie' (one row per game), 'rozklad_krytycznych' (how many games have
    0, 1, 2, ... critical positions, or 'unknown') and 'opis' (legend).
    Close the file in Excel before exporting, otherwise Windows blocks it."""
    reg = load_registry(registry_csv)
    dist = critical_distribution(reg)
    legend = pd.DataFrame({
        "kolumna": ["game_key", "n_critical", "complete", "analysed_positions",
                    "eligible_positions", "method"],
        "znaczenie": [
            "adres partii na Lichess (lub plik#indeks)",
            "liczba znalezionych pozycji krytycznych w partii",
            "True = przeanalizowano wszystkie pozycje partii, n_critical jest dokladne; "
            "False = analiza czesciowa, n_critical to dolne ograniczenie (nie wiadomo ile naprawde)",
            "ile pozycji partii przeanalizowano Stockfishem",
            "ile pozycji w partii nadawalo sie do analizy (od min_ply, >=2 legalne ruchy)",
            "ktora funkcja przeanalizowala partie (backfill = dopisane z wczesniejszych plikow)",
        ],
    })
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
        reg.to_excel(w, sheet_name="partie", index=False)
        dist.to_excel(w, sheet_name="rozklad_krytycznych", index=False)
        legend.to_excel(w, sheet_name="opis", index=False)
    print("zapisano:", xlsx_path)
