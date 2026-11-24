"""
timecontrol.py

The analysis CSVs written before the increment filter existed have no
time-control information. These helpers read it back from the PGN headers
(matching games by their index `game_id` in the PGN file, exactly as
registry.backfill_registry does) and write it into the CSV, so that the
analysis can filter on the increment (e.g. only games with +3 s per move,
which avoids time-trouble blunders).

    tag_time_control("../data/lichess....pgn.zst", "../data/critical_positions.csv")
    time_control_report(pd.read_csv("../data/critical_positions.csv"))
"""

from __future__ import annotations

import os
import shutil
from typing import Optional

import pandas as pd

from critical import parse_time_control
from registry import _iter_headers


def tag_time_control(pgn_path: str, csv_path: str, backup: bool = True) -> pd.DataFrame:
    """Add the columns `time_control`, `base_seconds`, `increment` to
    `csv_path` (in place; a copy is kept as <csv>.bak_tc). `pgn_path` MUST
    be the file the games were taken from. Safe to run again: existing
    values are recomputed."""
    df = pd.read_csv(csv_path)
    ids = set(df["game_id"].astype(int))
    max_id = max(ids)
    tc_by_id = {}
    for idx, headers in enumerate(_iter_headers(pgn_path)):
        if idx in ids:
            tc_by_id[idx] = headers.get("TimeControl")
        if idx >= max_id:
            break
    df = df.drop(columns=[c for c in ("time_control", "base_seconds", "increment") if c in df.columns])
    df["time_control"] = df["game_id"].map(tc_by_id)
    parsed = df["time_control"].map(parse_time_control)
    df["base_seconds"] = parsed.map(lambda t: t[0])
    df["increment"] = parsed.map(lambda t: t[1])
    if backup:
        shutil.copy(csv_path, csv_path + ".bak_tc")
    df.to_csv(csv_path, index=False)
    print(f"tagged {df['game_id'].nunique()} games, {int(df['time_control'].isna().sum())} rows without TimeControl")
    return df


def time_control_report(df: pd.DataFrame, top: int = 15) -> pd.DataFrame:
    """How many GAMES (not rows) there are per time control and per
    increment. Needs the columns added by tag_time_control (or by the
    analysis functions, which store them for new games)."""
    games = df.drop_duplicates("game_id")
    by_tc = games["time_control"].value_counts(dropna=False).rename("games").to_frame()
    by_tc["share"] = (by_tc["games"] / len(games)).round(3)
    print(f"{len(games)} games in total")
    print("\nby increment (seconds per move):")
    print(games["increment"].value_counts(dropna=False).sort_index().to_string())
    return by_tc.head(top)
