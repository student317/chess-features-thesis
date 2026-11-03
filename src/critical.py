"""
critical.py

Find "critical" positions with Stockfish and check whether the player
found the right move.

Definition used here
--------------------
A position is *critical* when there is essentially one good move: the best
move is much better than the second-best one. We ask Stockfish for its two
best moves (MultiPV=2) and compare their winning chances (not raw
centipawns, because 300 cp matters at 0.00 but not at +15.00):

    gap = win_prob(best) - win_prob(second_best)
    is_critical = gap >= gap_threshold      (default 0.20 = 20 percentage points)

Because the second-best move is, by definition, the best *alternative*,
in a critical position EVERY move other than the best one is at least
`gap` worse. So "player did not play the best move" == "player made a real
mistake", and we do not need to evaluate the played move separately.

Requires a Stockfish executable (https://stockfishchess.org/download/).
"""

from __future__ import annotations

import asyncio
import bisect
import io
import math
import os
import random
import sys
from typing import Iterator, Optional

import chess
import chess.engine
import chess.pgn
import pandas as pd
from tqdm import tqdm

from registry import append_registry, make_game_key, make_row, skip_keys

MATE_SCORE = 10000  # centipawn value used for forced mates


def win_prob(cp: float) -> float:
    """Winning chance in [0, 1] from centipawns (side-to-move point of view).
    Same logistic curve Lichess uses for its accuracy/win% numbers."""
    return 0.5 + 0.5 * (2 / (1 + math.exp(-0.00368208 * cp)) - 1)


def _read_all(handle) -> Iterator[chess.pgn.Game]:
    while True:
        game = chess.pgn.read_game(handle)
        if game is None:
            break
        yield game


def iter_games(path: str) -> Iterator[chess.pgn.Game]:
    """Iterate over games of a .pgn or .pgn.zst file (streaming, low memory)."""
    if path.endswith(".zst"):
        import zstandard as zstd

        with open(path, "rb") as fh:
            with zstd.ZstdDecompressor().stream_reader(fh) as reader:
                text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
                yield from _read_all(text)
    else:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            yield from _read_all(f)


def _int_or_none(x) -> Optional[int]:
    try:
        return int(x)
    except (ValueError, TypeError):
        return None


def analyse_game(
    engine: chess.engine.SimpleEngine,
    game: chess.pgn.Game,
    game_id: int,
    limit: chess.engine.Limit,
    gap_threshold: float = 0.20,
    min_ply: int = 8,
    multipv: int = 2,
) -> list[dict]:
    """Analyse every position of one game; return one row per analysed position."""
    white_elo = _int_or_none(game.headers.get("WhiteElo"))
    black_elo = _int_or_none(game.headers.get("BlackElo"))

    board = game.board()
    rows: list[dict] = []
    ply = 0
    for move in game.mainline_moves():
        ply += 1
        # skip the opening (book moves are not interesting) and forced moves
        n_legal = board.legal_moves.count()
        if ply >= min_ply and n_legal >= 2:
            infos = engine.analyse(board, limit, multipv=multipv)
            if len(infos) >= 2 and infos[0].get("pv") and infos[1].get("pv"):
                best_move = infos[0]["pv"][0]
                second_move = infos[1]["pv"][0]
                best_cp = infos[0]["score"].pov(board.turn).score(mate_score=MATE_SCORE)
                second_cp = infos[1]["score"].pov(board.turn).score(mate_score=MATE_SCORE)
                gap = win_prob(best_cp) - win_prob(second_cp)

                white_to_move = board.turn == chess.WHITE
                best_piece = board.piece_type_at(best_move.from_square)
                played_piece = board.piece_type_at(move.from_square)

                rows.append({
                    "game_id": game_id,
                    "ply": ply,
                    "fen": board.fen(),
                    "side_to_move": "white" if white_to_move else "black",
                    "player_elo": white_elo if white_to_move else black_elo,
                    "white_elo": white_elo,
                    "black_elo": black_elo,
                    "n_legal_moves": n_legal,
                    "best_move": best_move.uci(),
                    "second_move": second_move.uci(),
                    "played_move": move.uci(),
                    "best_cp": best_cp,
                    "second_cp": second_cp,
                    "gap_win_prob": gap,
                    "is_critical": gap >= gap_threshold,
                    "found_best": move == best_move,
                    "best_piece": chess.piece_name(best_piece) if best_piece else None,
                    "played_piece": chess.piece_name(played_piece) if played_piece else None,
                })
        board.push(move)
    return rows


def _game_elo(game: chess.pgn.Game) -> Optional[float]:
    """Mean of both players' Elo (None if either is missing)."""
    we = _int_or_none(game.headers.get("WhiteElo"))
    be = _int_or_none(game.headers.get("BlackElo"))
    if we is None or be is None:
        return None
    return (we + be) / 2


def _load_progress(output_csv: str, quota_edges: Optional[list[int]]):
    """Read an existing results file: which game_ids are done, and how many
    games each Elo bin already has (for quota sampling)."""
    done_ids: set[int] = set()
    bin_counts: dict[int, int] = {}
    if not os.path.exists(output_csv):
        return done_ids, bin_counts
    old = pd.read_csv(output_csv, usecols=lambda c: c in {"game_id", "white_elo", "black_elo", "player_elo"})
    done_ids = set(old["game_id"].unique())
    if quota_edges:
        per_game = old.groupby("game_id")["player_elo"].mean()   # fallback
        if {"white_elo", "black_elo"} <= set(old.columns):
            exact = old.groupby("game_id")[["white_elo", "black_elo"]].first().mean(axis=1, skipna=False)
            per_game = exact.fillna(per_game)   # old rows may have empty white/black_elo
        idx = pd.cut(per_game, bins=quota_edges, labels=False)
        bin_counts = idx.value_counts().to_dict()
        bin_counts = {int(k): int(v) for k, v in bin_counts.items() if pd.notna(k)}
    return done_ids, bin_counts


def find_critical_positions(
    pgn_path: str,
    engine_path: str,
    output_csv: str = "../data/critical_positions.csv",
    max_games: Optional[int] = 100,
    quota_edges: Optional[list[int]] = None,
    games_per_bin: Optional[int] = None,
    depth: int = 12,
    gap_threshold: float = 0.20,
    min_ply: int = 8,
    min_elo: Optional[int] = None,
    threads: int = 2,
    hash_mb: int = 128,
    multipv: int = 2,
    save_every: int = 10,
    sample_prob: float = 1.0,
    seed: int = 42,
    registry_path: Optional[str] = None,
    registry_skip: str = "all",
) -> pd.DataFrame:
    """
    Run Stockfish over games from a PGN / PGN.ZST file and APPEND results to
    `output_csv` every `save_every` games. Returns the whole results file.

    * RESUMABLE: if `output_csv` already exists, games already in it are
      skipped. Interrupt the cell (Stop button) any time and run it again -
      nothing is lost, it continues where it stopped.
    * max_games=None  -> go through the entire file (until the end or until
      all quotas are full). NOTE: a Lichess monthly dump has tens of millions
      of games - that is not feasible, use a limit or quotas.
    * quotas: pass quota_edges=[0,1200,1600,2000,4000] and games_per_bin=500
      to collect up to 500 games in each Elo bin (by mean Elo of the two
      players). Stops when every bin is full. Gives balanced data even though
      most Lichess players are ~1500-1800.

    * sample_prob < 1.0 -> each game is taken only with this probability
      (e.g. 0.01 = ~1% of games), so the sample is spread over the whole file
      instead of being just its first games. Slower to fill, because the
      file has to be read much further.

    The Elo bins you use for ANALYSIS are independent of this: results are
    stored with raw `player_elo`, so re-binning needs no re-analysis.
    """
    limit = chess.engine.Limit(depth=depth)
    done_ids, bin_counts = _load_progress(output_csv, quota_edges)
    n_bins = (len(quota_edges) - 1) if quota_edges else 0
    use_quota = bool(quota_edges and games_per_bin)

    def quotas_full() -> bool:
        return use_quota and all(bin_counts.get(b, 0) >= games_per_bin for b in range(n_bins))

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    buffer: list[dict] = []
    reg_buffer: list[dict] = []
    rng = random.Random(seed)
    reg_keys = skip_keys(registry_path, registry_skip)   # None when no registry

    def flush():
        if buffer:
            chunk = pd.DataFrame(buffer)
            if os.path.exists(output_csv):
                # keep the column layout of the existing file, so appending can
                # never produce rows with a different number of fields
                existing_cols = list(pd.read_csv(output_csv, nrows=0).columns)
                chunk = chunk.reindex(columns=existing_cols)
                chunk.to_csv(output_csv, mode="a", index=False, header=False)
            else:
                chunk.to_csv(output_csv, index=False)
            buffer.clear()
        if registry_path and reg_buffer:      # only after the results are on disk
            append_registry(registry_path, reg_buffer)
            reg_buffer.clear()

    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    try:
        engine.configure({"Threads": threads, "Hash": hash_mb})
        total = max_games if max_games is not None else None
        pbar = tqdm(total=total, desc="games analysed", initial=len(done_ids))
        done = len(done_ids)
        since_flush = 0
        for game_id, game in enumerate(iter_games(pgn_path)):
            if max_games is not None and done >= max_games:
                break
            if quotas_full():
                break
            if game_id in done_ids:
                continue
            key = make_game_key(game.headers, pgn_path, game_id)
            if reg_keys is not None and key in reg_keys:
                continue                      # already analysed (registry)
            if sample_prob < 1.0 and rng.random() > sample_prob:
                continue
            g_elo = _game_elo(game)
            if min_elo is not None and (g_elo is None or g_elo < min_elo):
                continue
            if use_quota:
                if g_elo is None:
                    continue
                b = pd.cut([g_elo], bins=quota_edges, labels=False)[0]
                if pd.isna(b) or bin_counts.get(int(b), 0) >= games_per_bin:
                    continue
                bin_counts[int(b)] = bin_counts.get(int(b), 0) + 1
            rows = analyse_game(engine, game, game_id, limit, gap_threshold, min_ply, multipv)
            buffer.extend(rows)
            if registry_path:
                reg_buffer.append(make_row(
                    key, pgn_path, game_id,
                    _int_or_none(game.headers.get("WhiteElo")),
                    _int_or_none(game.headers.get("BlackElo")),
                    len(rows), len(rows), sum(bool(r["is_critical"]) for r in rows),
                    True, "find_critical_positions", depth, gap_threshold, min_ply))
            done += 1
            since_flush += 1
            pbar.update(1)
            if since_flush >= save_every:
                flush()
                since_flush = 0
        pbar.close()
    finally:
        flush()          # also runs when you press Stop / an error happens
        engine.quit()

    return pd.read_csv(output_csv) if os.path.exists(output_csv) else pd.DataFrame()


# ---------------------------------------------------------------------------
# Targeted collection: keep searching until we have N critical positions
# ---------------------------------------------------------------------------

def analyse_position(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    limit: chess.engine.Limit,
    gap_threshold: float = 0.20,
    multipv: int = 2,
) -> Optional[dict]:
    """Analyse ONE position. Returns None if Stockfish gave < 2 moves."""
    infos = engine.analyse(board, limit, multipv=multipv)
    if len(infos) < 2 or not infos[0].get("pv") or not infos[1].get("pv"):
        return None
    best_move = infos[0]["pv"][0]
    second_move = infos[1]["pv"][0]
    best_cp = infos[0]["score"].pov(board.turn).score(mate_score=MATE_SCORE)
    second_cp = infos[1]["score"].pov(board.turn).score(mate_score=MATE_SCORE)
    gap = win_prob(best_cp) - win_prob(second_cp)
    piece = board.piece_type_at(best_move.from_square)
    return {
        "best_move": best_move,
        "second_move": second_move,
        "best_cp": best_cp,
        "second_cp": second_cp,
        "gap_win_prob": gap,
        "is_critical": gap >= gap_threshold,
        "best_piece": chess.piece_name(piece) if piece else None,
    }


def _append_rows(output_csv: str, rows: list[dict]) -> None:
    """Append rows to a CSV, keeping the existing file's column layout."""
    if not rows:
        return
    chunk = pd.DataFrame(rows)
    if os.path.exists(output_csv):
        cols = list(pd.read_csv(output_csv, nrows=0).columns)
        chunk.reindex(columns=cols).to_csv(output_csv, mode="a", index=False, header=False)
    else:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        chunk.to_csv(output_csv, index=False)


def collect_critical_positions(
    pgn_path: str,
    engine_path: str,
    output_csv: str = "../data/critical_random.csv",
    target_per_bin: int = 100,
    elo_edges: Optional[list[int]] = None,
    depth: int = 12,
    gap_threshold: float = 0.20,
    min_ply: int = 8,
    max_critical_per_game: int = 2,
    max_positions_per_game: Optional[int] = 40,
    sample_prob: float = 1.0,
    seed: int = 42,
    max_games_scanned: Optional[int] = None,
    min_elo: Optional[int] = None,
    threads: int = 2,
    hash_mb: int = 128,
    registry_path: Optional[str] = None,
    registry_skip: str = "all",
) -> pd.DataFrame:
    """
    Keep searching games until we have `target_per_bin` CRITICAL positions
    in every Elo bin (bins are by the Elo of the player who had to move).

    * Positions inside a game are checked in RANDOM order; a game gives at
      most `max_critical_per_game` positions (so one game cannot dominate).
    * A game with no critical position simply gives nothing - the search
      moves on to the next game. It never stops early because of that.
    * Positions whose Elo bin is already full are not even analysed.
    * elo_edges=None -> one single bin, `target_per_bin` is then the total.
    * sample_prob < 1 -> only a random fraction of games is considered, so
      the sample is spread over the file (slower to reach the target).
    * Resumable: existing rows in `output_csv` count toward the targets and
      their games are skipped. Stop the cell any time, run it again later.
    * Only CRITICAL positions are saved here, so this output cannot tell you
      "what share of positions is critical" - use find_critical_positions
      for that. It is meant for: "of critical positions, how often is the
      best move found, by Elo and piece".
    """
    rng = random.Random(seed)
    reg_keys = skip_keys(registry_path, registry_skip)   # None when no registry
    limit = chess.engine.Limit(depth=depth)
    n_bins = (len(elo_edges) - 1) if elo_edges else 1

    def bin_of(elo) -> Optional[int]:
        if elo is None or (isinstance(elo, float) and math.isnan(elo)):
            return None
        if not elo_edges:
            return 0
        b = bisect.bisect_left(elo_edges, elo) - 1      # bins are (a, b]
        return b if 0 <= b < n_bins else None

    counts = {b: 0 for b in range(n_bins)}
    done_ids: set[int] = set()
    if os.path.exists(output_csv):
        old = pd.read_csv(output_csv, usecols=["game_id", "player_elo"])
        done_ids = set(old["game_id"].unique())
        for e in old["player_elo"]:
            b = bin_of(e)
            if b is not None:
                counts[b] += 1

    def all_full() -> bool:
        return all(counts[b] >= target_per_bin for b in range(n_bins))

    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    try:
        engine.configure({"Threads": threads, "Hash": hash_mb})
        total_target = n_bins * target_per_bin
        pbar = tqdm(total=total_target, initial=min(sum(counts.values()), total_target),
                    desc="critical positions found")
        scanned = 0
        for game_id, game in enumerate(iter_games(pgn_path)):
            if all_full():
                break
            if max_games_scanned is not None and scanned >= max_games_scanned:
                break
            if game_id in done_ids:
                continue
            key = make_game_key(game.headers, pgn_path, game_id)
            if reg_keys is not None and key in reg_keys:
                continue                      # already analysed (registry)
            if sample_prob < 1.0 and rng.random() > sample_prob:
                continue
            we = _int_or_none(game.headers.get("WhiteElo"))
            be = _int_or_none(game.headers.get("BlackElo"))
            if min_elo is not None and (we is None or be is None or min(we, be) < min_elo):
                continue
            scanned += 1

            # collect candidate positions of this game
            candidates = []
            eligible_total = 0     # positions that could be analysed in this game
            board = game.board()
            ply = 0
            for move in game.mainline_moves():
                ply += 1
                if ply >= min_ply:
                    n_legal = board.legal_moves.count()
                    if n_legal >= 2:
                        eligible_total += 1
                        white = board.turn == chess.WHITE
                        elo = we if white else be
                        b = bin_of(elo)
                        if b is not None and counts[b] < target_per_bin:
                            candidates.append((ply, board.copy(), move, elo, b, white, n_legal))
                board.push(move)

            rng.shuffle(candidates)
            if max_positions_per_game is not None:
                candidates = candidates[:max_positions_per_game]

            rows, found = [], 0
            analysed = n_crit_seen = 0
            for ply, b_board, played, elo, b, white, n_legal in candidates:
                if counts[b] >= target_per_bin:
                    continue
                res = analyse_position(engine, b_board, limit, gap_threshold)
                analysed += 1
                if res is None or not res["is_critical"]:
                    continue
                n_crit_seen += 1
                played_piece = b_board.piece_type_at(played.from_square)
                rows.append({
                    "game_id": game_id,
                    "ply": ply,
                    "fen": b_board.fen(),
                    "side_to_move": "white" if white else "black",
                    "player_elo": elo,
                    "white_elo": we,
                    "black_elo": be,
                    "n_legal_moves": n_legal,
                    "best_move": res["best_move"].uci(),
                    "second_move": res["second_move"].uci(),
                    "played_move": played.uci(),
                    "best_cp": res["best_cp"],
                    "second_cp": res["second_cp"],
                    "gap_win_prob": res["gap_win_prob"],
                    "is_critical": True,
                    "found_best": played == res["best_move"],
                    "best_piece": res["best_piece"],
                    "played_piece": chess.piece_name(played_piece) if played_piece else None,
                })
                counts[b] += 1
                found += 1
                pbar.update(1)
                if found >= max_critical_per_game:
                    break
            _append_rows(output_csv, rows)
            if registry_path:
                append_registry(registry_path, [make_row(
                    key, pgn_path, game_id, we, be, analysed, eligible_total, n_crit_seen,
                    analysed == eligible_total, "collect_critical", depth, gap_threshold, min_ply)])
        pbar.close()
    finally:
        engine.quit()

    print("found per bin:", counts, "| games scanned this run:", scanned)
    return pd.read_csv(output_csv) if os.path.exists(output_csv) else pd.DataFrame()


# ---------------------------------------------------------------------------
# Balanced collection: the same number of critical positions per best piece
# ---------------------------------------------------------------------------

PIECE_NAMES = ["pawn", "knight", "bishop", "rook", "queen", "king"]


def collect_critical_by_piece(
    pgn_path: str,
    engine_path: str,
    output_csv: str = "../data/critical_by_piece.csv",
    target_per_piece: int = 100,
    elo_edges: Optional[list[int]] = None,
    pieces: Optional[list[str]] = None,
    depth: int = 12,
    gap_threshold: float = 0.20,
    min_ply: int = 8,
    max_critical_per_game: int = 2,
    max_positions_per_game: Optional[int] = 40,
    sample_prob: float = 1.0,
    seed: int = 42,
    max_games_scanned: Optional[int] = None,
    min_elo: Optional[int] = None,
    threads: int = 2,
    hash_mb: int = 128,
    registry_path: Optional[str] = None,
    registry_skip: str = "all",
) -> pd.DataFrame:
    """
    Collect CRITICAL positions so that every piece type that makes the best
    move (pawn, knight, bishop, rook, queen, king) has the same number of
    them: `target_per_piece`.

    * elo_edges=None  -> `target_per_piece` per piece, regardless of Elo.
    * elo_edges=[...] -> `target_per_piece` per piece AND per Elo bin
      (so 6 pieces x 4 bins x target positions - this takes much longer).
    * Which piece makes the best move is only known AFTER Stockfish
      analysed the position, so a position can be analysed and then dropped
      because its cell is already full.
    * Some pieces are rare as the only good move (queen, bishop, king in
      the middlegame) - the search may need many games. You can stop the
      cell any time; it is resumable (existing rows count toward targets).
    * Because pieces are balanced artificially, the overall found-best rate
      across all pieces is NOT representative. Compare rates piece by
      piece (and by Elo), do not average them.
    """
    pieces = pieces or PIECE_NAMES
    rng = random.Random(seed)
    reg_keys = skip_keys(registry_path, registry_skip)   # None when no registry
    limit = chess.engine.Limit(depth=depth)
    n_bins = (len(elo_edges) - 1) if elo_edges else 1

    def bin_of(elo) -> Optional[int]:
        if elo is None or (isinstance(elo, float) and math.isnan(elo)):
            return None
        if not elo_edges:
            return 0
        b = bisect.bisect_left(elo_edges, elo) - 1      # bins are (a, b]
        return b if 0 <= b < n_bins else None

    counts = {(p, b): 0 for p in pieces for b in range(n_bins)}
    done_ids: set[int] = set()
    if os.path.exists(output_csv):
        old = pd.read_csv(output_csv, usecols=["game_id", "player_elo", "best_piece"])
        done_ids = set(old["game_id"].unique())
        for p, e in zip(old["best_piece"], old["player_elo"]):
            b = bin_of(e)
            if b is not None and (p, b) in counts:
                counts[(p, b)] += 1

    def all_full() -> bool:
        return all(c >= target_per_piece for c in counts.values())

    def bin_has_room(b: int) -> bool:
        return any(counts[(p, b)] < target_per_piece for p in pieces)

    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    scanned = 0
    try:
        engine.configure({"Threads": threads, "Hash": hash_mb})
        total_target = len(counts) * target_per_piece
        pbar = tqdm(total=total_target, initial=min(sum(counts.values()), total_target),
                    desc="critical positions found")
        for game_id, game in enumerate(iter_games(pgn_path)):
            if all_full():
                break
            if max_games_scanned is not None and scanned >= max_games_scanned:
                break
            if game_id in done_ids:
                continue
            key = make_game_key(game.headers, pgn_path, game_id)
            if reg_keys is not None and key in reg_keys:
                continue                      # already analysed (registry)
            if sample_prob < 1.0 and rng.random() > sample_prob:
                continue
            we = _int_or_none(game.headers.get("WhiteElo"))
            be = _int_or_none(game.headers.get("BlackElo"))
            if min_elo is not None and (we is None or be is None or min(we, be) < min_elo):
                continue
            scanned += 1

            candidates = []
            eligible_total = 0     # positions that could be analysed in this game
            board = game.board()
            ply = 0
            for move in game.mainline_moves():
                ply += 1
                if ply >= min_ply:
                    n_legal = board.legal_moves.count()
                    if n_legal >= 2:
                        eligible_total += 1
                        white = board.turn == chess.WHITE
                        elo = we if white else be
                        b = bin_of(elo)
                        if b is not None and bin_has_room(b):
                            candidates.append((ply, board.copy(), move, elo, b, white, n_legal))
                board.push(move)

            rng.shuffle(candidates)
            if max_positions_per_game is not None:
                candidates = candidates[:max_positions_per_game]

            rows, found = [], 0
            analysed = n_crit_seen = 0
            for ply, b_board, played, elo, b, white, n_legal in candidates:
                if not bin_has_room(b):
                    continue
                res = analyse_position(engine, b_board, limit, gap_threshold)
                analysed += 1
                if res is None or not res["is_critical"]:
                    continue
                n_crit_seen += 1
                key = (res["best_piece"], b)
                if key not in counts or counts[key] >= target_per_piece:
                    continue                      # this piece/bin is already full
                played_piece = b_board.piece_type_at(played.from_square)
                rows.append({
                    "game_id": game_id,
                    "ply": ply,
                    "fen": b_board.fen(),
                    "side_to_move": "white" if white else "black",
                    "player_elo": elo,
                    "white_elo": we,
                    "black_elo": be,
                    "n_legal_moves": n_legal,
                    "best_move": res["best_move"].uci(),
                    "second_move": res["second_move"].uci(),
                    "played_move": played.uci(),
                    "best_cp": res["best_cp"],
                    "second_cp": res["second_cp"],
                    "gap_win_prob": res["gap_win_prob"],
                    "is_critical": True,
                    "found_best": played == res["best_move"],
                    "best_piece": res["best_piece"],
                    "played_piece": chess.piece_name(played_piece) if played_piece else None,
                })
                counts[key] += 1
                found += 1
                pbar.update(1)
                if found >= max_critical_per_game:
                    break
            _append_rows(output_csv, rows)
            if registry_path:
                append_registry(registry_path, [make_row(
                    key, pgn_path, game_id, we, be, analysed, eligible_total, n_crit_seen,
                    analysed == eligible_total, "collect_critical", depth, gap_threshold, min_ply)])
        pbar.close()
    finally:
        engine.quit()

    table = pd.Series(counts).unstack(fill_value=0) if n_bins > 1 else pd.Series(
        {p: counts[(p, 0)] for p in pieces})
    print("found per piece (rows) / Elo bin (columns):")
    print(table)
    print("games scanned this run:", scanned)
    return pd.read_csv(output_csv) if os.path.exists(output_csv) else pd.DataFrame()
