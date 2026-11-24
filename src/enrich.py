"""
enrich.py

Second analysis stage on top of critical_positions.csv (Stockfish needed):

1. enrich_critical(...)       for every CRITICAL position
       missed_*   tactics realised by the BEST move  (what the player could have played)
       allowed_*  tactics the OPPONENT gets after the player's actual mistake
                  (only for mistakes; needs the opponent's best reply)
       pv / opp_pv  the engine lines used (needed by 'elimination of the defence')
       tactic_length (optional)  number of consecutive critical positions for
                  the player along the engine line (1 = single only-move)

2. find_unforced_errors(...)  for NON-critical positions (a second good option
       exists, gap < threshold): did the player still pick a move that loses
       more than `blunder_threshold` winning chance? For those, the same tactic
       labels are computed.

Both are resumable (results are appended to a CSV, rows already there are
skipped), take an `increment` filter (needs the `increment` column - see
timecontrol.tag_time_control) and cost roughly one engine search per row.
"""

from __future__ import annotations

import os
from typing import Optional

import chess
import chess.engine
import pandas as pd
from tqdm import tqdm

import tactics as T
from critical import MATE_SCORE, _append_rows, win_prob

# tactics that make sense as "what the OPPONENT gets" after our mistake
ATTACKING = [T.FORCED_MATE, T.DOUBLE_ATTACK, T.PIN, T.DISCOVERED_ATTACK, T.ELIMINATION,
             T.FAVOURABLE_EXCHANGE, T.CAPTURE_ATTACKED_MORE, T.CAPTURE_UNPROTECTED]

# when one move matches several tactics, the first one in this list is the "primary" label
PRIORITY = [T.FORCED_MATE, T.DEFEND_AGAINST_MATE, T.DOUBLE_ATTACK, T.PIN, T.DISCOVERED_ATTACK,
            T.ELIMINATION, T.FAVOURABLE_EXCHANGE, T.CAPTURE_ATTACKED_MORE, T.CAPTURE_UNPROTECTED,
            T.DEFEND_THREATENED]


def primary(tactics: list) -> str:
    for t in PRIORITY:
        if t in tactics:
            return t
    return "none"


def _truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true")


def _cp(score, turn) -> Optional[int]:
    return None if score is None else score.pov(turn).score(mate_score=MATE_SCORE)


def _uci(moves) -> str:
    return " ".join(m.uci() for m in moves)


def tactic_length(engine, board: chess.Board, limit, gap_threshold: float = 0.20, max_len: int = 5) -> int:
    """Number of consecutive CRITICAL positions for the player along the
    engine's main line, starting at `board`: position critical -> play the
    engine's best move and the engine's best reply -> is the next position
    critical again? ... (up to max_len). 0 when even `board` is not critical
    at this search (noise), positions with a single legal move end the count."""
    b = board.copy(stack=False)
    length = 0
    while length < max_len:
        infos = engine.analyse(b, limit, multipv=2)
        if len(infos) < 2 or not infos[0].get("pv") or not infos[1].get("pv"):
            break
        best = _cp(infos[0]["score"], b.turn)
        second = _cp(infos[1]["score"], b.turn)
        if win_prob(best) - win_prob(second) < gap_threshold:
            break
        length += 1
        pv = infos[0]["pv"]
        b.push(pv[0])
        if b.is_game_over() or len(pv) < 2:
            break
        b.push(pv[1])
        if b.is_game_over():
            break
    return length


def _after_move_analysis(engine, board: chess.Board, move: chess.Move, limit):
    """Analyse the position after `move`. Returns (board_after, pv list,
    score from the point of view of the side to move there)."""
    b = board.copy(stack=False)
    b.push(move)
    if b.is_game_over():
        return b, [], None
    info = engine.analyse(b, limit)
    return b, list(info.get("pv", [])), _cp(info.get("score"), b.turn)


def _labels(detections: list) -> tuple:
    ids = [d.tactic for d in detections]
    return ";".join(ids), primary(ids), " | ".join(f"{d.tactic}: {d.note}" for d in detections)


def classify_row(engine, limit, row: dict, compute_length: bool = False, gap_threshold: float = 0.20,
                 max_len: int = 5, with_missed_pv: bool = True) -> dict:
    """Tactic labels for one analysed position (a row of critical_positions.csv)."""
    board = chess.Board(row["fen"])
    best = chess.Move.from_uci(row["best_move"])
    played = chess.Move.from_uci(row["played_move"])
    found = bool(row["found_best"])

    # --- missed: the best move; its line (best, reply, our next move) for 'elimination'
    pv_best = [best]
    if with_missed_pv:
        _, line, _ = _after_move_analysis(engine, board, best, limit)
        pv_best += line[:2]
    missed = T.detect_all(board, best, pv=pv_best if len(pv_best) >= 3 else None,
                          best_cp=row["best_cp"], second_cp=row["second_cp"])
    m_ids, m_primary, m_notes = _labels(missed)

    # --- allowed: what the opponent gets after the played move (mistakes only)
    a_ids = a_primary = a_notes = ""
    opp_reply = opp_pv = ""
    opp_cp = None
    played_cp = None
    if not found:
        b2, line, opp_cp = _after_move_analysis(engine, board, played, limit)
        if line:
            opp_reply = line[0].uci()
            opp_pv = _uci(line[:3])
            allowed = [d for d in T.detect_all(b2, line[0], only=ATTACKING, pv=line[:3] if len(line) >= 3 else None, best_cp=opp_cp)
                       if d.tactic in ATTACKING]
            a_ids, a_primary, a_notes = _labels(allowed)
        if opp_cp is not None:
            played_cp = -opp_cp                    # from the player's point of view

    out = {
        "missed_tactics": m_ids, "missed_primary": m_primary, "missed_notes": m_notes,
        "allowed_tactics": a_ids, "allowed_primary": a_primary or ("none" if not found else ""),
        "allowed_notes": a_notes,
        "pv_best": _uci(pv_best), "opp_reply": opp_reply, "opp_pv": opp_pv,
        "opp_cp": opp_cp, "played_cp": played_cp,
    }
    # whose tactic is it? player = best move would realise it (missed),
    # opponent = the played move gives it away (allowed)
    out["tactic_owner"] = ("both" if m_ids and a_ids else "player" if m_ids
                           else "opponent" if a_ids else "none")
    if compute_length:
        out["tactic_length"] = tactic_length(engine, board, limit, gap_threshold, max_len)
    return out


def _prepare(main_csv: str, out_csv: str, critical: bool, increment: Optional[int], min_elo: Optional[int],
             max_rows: Optional[int], seed: int, sample_n: Optional[int] = None,
             bins: Optional[list] = None, per_bin: Optional[int] = None) -> pd.DataFrame:
    df = pd.read_csv(main_csv)
    df["is_critical"] = _truthy(df["is_critical"])
    df["found_best"] = _truthy(df["found_best"])
    df = df[df["is_critical"]] if critical else df[~df["is_critical"]]
    if increment is not None:
        if "increment" not in df.columns:
            raise ValueError("no 'increment' column - run timecontrol.tag_time_control(pgn, csv) first")
        df = df[df["increment"] == increment]
    if min_elo is not None:
        df = df[df["player_elo"] >= min_elo]
    if sample_n is not None and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=seed)
    if per_bin is not None:
        # same number of positions in every Elo bin of the PLAYER (bins like in the plots)
        if bins is None:
            raise ValueError("per_bin needs the Elo bin edges: bins=BINS")
        df = df.assign(_bin=pd.cut(df["player_elo"], bins=bins))
        parts = [g.sample(n=min(per_bin, len(g)), random_state=seed) for _, g in df.groupby("_bin", observed=True)]
        df = pd.concat(parts).drop(columns="_bin")
    if os.path.exists(out_csv):
        done = pd.read_csv(out_csv, usecols=["game_id", "ply"])
        keys = set(zip(done["game_id"], done["ply"]))
        df = df[[(g, p) not in keys for g, p in zip(df["game_id"], df["ply"])]]
    df = df.sample(frac=1, random_state=seed)          # random order: a partial run is a random sample
    if max_rows is not None:
        df = df.head(max_rows)
    return df


def enrich_critical(
    main_csv: str,
    out_csv: str,
    engine_path: str,
    depth: int = 12,
    increment: Optional[int] = None,
    min_elo: Optional[int] = None,
    max_rows: Optional[int] = None,
    compute_length: bool = False,
    gap_threshold: float = 0.20,
    threads: int = 2,
    hash_mb: int = 128,
    save_every: int = 25,
    seed: int = 0,
) -> pd.DataFrame:
    """Tactic labels for the critical positions of `main_csv` (see module
    docstring). ~1 search per correct decision and 2 per mistake (3-5 more
    with compute_length). Resumable: stop the cell any time and run it again.
    max_rows limits the work of THIS run (random order)."""
    todo = _prepare(main_csv, out_csv, True, increment, min_elo, max_rows, seed)
    print(f"{len(todo)} critical positions to process")
    limit = chess.engine.Limit(depth=depth)
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    buffer: list = []
    try:
        engine.configure({"Threads": threads, "Hash": hash_mb})
        for row in tqdm(todo.to_dict("records"), desc="critical positions"):
            res = classify_row(engine, limit, row, compute_length, gap_threshold)
            buffer.append({"game_id": row["game_id"], "ply": row["ply"], "player_elo": row["player_elo"],
                           "increment": row.get("increment"), "found_best": row["found_best"], **res})
            if len(buffer) >= save_every:
                _append_rows(out_csv, buffer)
                buffer.clear()
    finally:
        _append_rows(out_csv, buffer)
        engine.quit()
    return pd.read_csv(out_csv) if os.path.exists(out_csv) else pd.DataFrame()


def add_tactic_length(
    main_csv: str,
    enriched_csv: str,
    engine_path: str,
    depth: int = 12,
    max_rows: Optional[int] = None,
    only_labelled: bool = True,
    gap_threshold: float = 0.20,
    max_len: int = 5,
    threads: int = 2,
    hash_mb: int = 128,
    save_every: int = 25,
    seed: int = 0,
) -> pd.DataFrame:
    """Fill the `tactic_length` column of an EXISTING enriched file (rows done
    earlier with compute_length=False), without recomputing the labels.
    Only rows with `tactic_length` still empty are processed; with
    only_labelled=True only rows that have a missed or allowed tactic
    (the ones Bart's classification is about). Resumable (the file is
    rewritten every `save_every` rows; a backup .bak_len is made once).
    A row costs 1-5 engine searches (MultiPV=2). max_rows limits THIS run."""
    import shutil

    bak = enriched_csv + ".bak_len"
    if not os.path.exists(bak):
        shutil.copyfile(enriched_csv, bak)
    en = pd.read_csv(enriched_csv)
    if "tactic_length" not in en.columns:
        en["tactic_length"] = float("nan")
    fen = pd.read_csv(main_csv, usecols=["game_id", "ply", "fen"]).drop_duplicates(["game_id", "ply"])
    work = en.merge(fen, on=["game_id", "ply"], how="left")
    mask = work["tactic_length"].isna() & work["fen"].notna()
    if only_labelled:
        mask &= work["missed_tactics"].fillna("").ne("") | work["allowed_tactics"].fillna("").ne("")
    todo = work[mask]
    if max_rows is not None and len(todo) > max_rows:
        todo = todo.sample(n=max_rows, random_state=seed)
    print(f"{len(todo)} rows to process")
    if todo.empty:
        return en
    limit = chess.engine.Limit(depth=depth)
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    done = 0
    try:
        engine.configure({"Threads": threads, "Hash": hash_mb})
        for idx, fen_str in tqdm(zip(todo.index, todo["fen"]), total=len(todo), desc="tactic length"):
            en.loc[idx, "tactic_length"] = tactic_length(engine, chess.Board(fen_str), limit, gap_threshold, max_len)
            done += 1
            if done % save_every == 0:
                en.to_csv(enriched_csv, index=False)
    finally:
        en.to_csv(enriched_csv, index=False)
        engine.quit()
    return en


def find_unforced_errors(
    main_csv: str,
    out_csv: str,
    engine_path: str,
    depth: int = 12,
    blunder_threshold: float = 0.20,
    increment: Optional[int] = None,
    min_elo: Optional[int] = None,
    sample_n: Optional[int] = None,
    max_rows: Optional[int] = None,
    threads: int = 2,
    hash_mb: int = 128,
    save_every: int = 25,
    seed: int = 0,
    bins: Optional[list] = None,
    per_bin: Optional[int] = None,
) -> pd.DataFrame:
    """Unforced errors: NON-critical positions (a second good option exists)
    where the player nevertheless played a move that loses at least
    `blunder_threshold` winning chance compared with the best move.

    * Moves equal to the best or second-best move cannot be such errors
      (the second-best is within the threshold by definition) - they are
      recorded without an engine search.
    * For every other move the position after it is searched once; its score
      gives the loss, its principal variation the opponent's reply.
    * For errors the tactics are labelled like in enrich_critical
      (missed = what the best move would have done, allowed = what the
      opponent gets). No pv for the best move here (no 'elimination' for
      missed tactics).
    * sample_n: random sample of non-critical positions, so that the error
      RATE (errors / rows in the output) stays unbiased and cheaper to get.
    * per_bin (+ bins=BINS): the SAME number of non-critical positions in every
      Elo bin of the player (bins smaller than per_bin give all they have).
      Use instead of sample_n; the rate per bin stays unbiased.
    """
    todo = _prepare(main_csv, out_csv, False, increment, min_elo, max_rows, seed, sample_n=sample_n,
                    bins=bins, per_bin=per_bin)
    print(f"{len(todo)} non-critical positions to process")
    limit = chess.engine.Limit(depth=depth)
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    buffer: list = []
    try:
        engine.configure({"Threads": threads, "Hash": hash_mb})
        for row in tqdm(todo.to_dict("records"), desc="non-critical positions"):
            out = {"game_id": row["game_id"], "ply": row["ply"], "player_elo": row["player_elo"],
                   "increment": row.get("increment"), "played_move": row["played_move"],
                   "best_move": row["best_move"], "second_move": row["second_move"],
                   "best_cp": row["best_cp"], "evaluated": False, "played_cp": None,
                   "loss_win_prob": None, "is_unforced_error": False,
                   "missed_tactics": "", "missed_primary": "", "allowed_tactics": "", "allowed_primary": "",
                   "opp_reply": "", "opp_pv": ""}
            if row["played_move"] not in (row["best_move"], row["second_move"]):
                board = chess.Board(row["fen"])
                played = chess.Move.from_uci(row["played_move"])
                b2, line, opp_cp = _after_move_analysis(engine, board, played, limit)
                if opp_cp is not None:
                    played_cp = -opp_cp
                    loss = win_prob(row["best_cp"]) - win_prob(played_cp)
                    out.update({"evaluated": True, "played_cp": played_cp, "loss_win_prob": loss,
                                "is_unforced_error": bool(loss >= blunder_threshold)})
                    if out["is_unforced_error"]:
                        best = chess.Move.from_uci(row["best_move"])
                        missed = T.detect_all(board, best, best_cp=row["best_cp"], second_cp=row["second_cp"])
                        out["missed_tactics"], out["missed_primary"], _ = _labels(missed)
                        if line:
                            allowed = [d for d in T.detect_all(b2, line[0], only=ATTACKING, pv=line[:3] if len(line) >= 3 else None,
                                                               best_cp=opp_cp) if d.tactic in ATTACKING]
                            out["allowed_tactics"], out["allowed_primary"], _ = _labels(allowed)
                            out["opp_reply"], out["opp_pv"] = line[0].uci(), _uci(line[:3])
            buffer.append(out)
            if len(buffer) >= save_every:
                _append_rows(out_csv, buffer)
                buffer.clear()
    finally:
        _append_rows(out_csv, buffer)
        engine.quit()
    return pd.read_csv(out_csv) if os.path.exists(out_csv) else pd.DataFrame()
