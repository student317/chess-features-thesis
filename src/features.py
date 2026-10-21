"""
features.py

Core feature-extraction logic: given a chess.Board (and the move that led
to it), compute a dict of features about the position / move.

Design goal: every feature is a small, independent function that takes
(board_before, move, board_after) and returns one or more named values.
This makes it trivial to add new features later (e.g. from Yngvi) without
touching the parsing/iteration code.

Usage:
    from features import process_pgn_file
    df = process_pgn_file("data/sample.pgn", max_games=1000)
    df.to_parquet("data/features.parquet")
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Callable, Optional

import chess
import chess.pgn
import pandas as pd
from tqdm import tqdm

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


# ---------------------------------------------------------------------------
# Individual feature functions.
# Each takes (board_before: chess.Board, move: chess.Move, board_after: chess.Board)
# and returns a dict of {feature_name: value}. Keep them small and independent.
# ---------------------------------------------------------------------------

def feat_material(board_before: chess.Board, move: chess.Move, board_after: chess.Board) -> dict:
    """Material balance (white - black) after the move, in pawns."""
    balance = 0
    for piece_type, value in PIECE_VALUES.items():
        balance += value * len(board_after.pieces(piece_type, chess.WHITE))
        balance -= value * len(board_after.pieces(piece_type, chess.BLACK))
    return {"material_balance": balance}


def feat_bishop_pair(board_before: chess.Board, move: chess.Move, board_after: chess.Board) -> dict:
    return {
        "white_has_bishop_pair": len(board_after.pieces(chess.BISHOP, chess.WHITE)) >= 2,
        "black_has_bishop_pair": len(board_after.pieces(chess.BISHOP, chess.BLACK)) >= 2,
    }


def feat_castling_rights(board_before: chess.Board, move: chess.Move, board_after: chess.Board) -> dict:
    return {
        "white_kingside_rights": board_after.has_kingside_castling_rights(chess.WHITE),
        "white_queenside_rights": board_after.has_queenside_castling_rights(chess.WHITE),
        "black_kingside_rights": board_after.has_kingside_castling_rights(chess.BLACK),
        "black_queenside_rights": board_after.has_queenside_castling_rights(chess.BLACK),
        "is_castling_move": board_before.is_castling(move),
    }


def feat_check(board_before: chess.Board, move: chess.Move, board_after: chess.Board) -> dict:
    return {"gives_check": board_after.is_check()}


def feat_en_passant(board_before: chess.Board, move: chess.Move, board_after: chess.Board) -> dict:
    return {
        "is_en_passant_capture": board_before.is_en_passant(move),
        # was en passant even legally available in this position, whether or not it was taken
        "en_passant_was_available": board_before.has_legal_en_passant(),
    }


def feat_promotion(board_before: chess.Board, move: chess.Move, board_after: chess.Board) -> dict:
    promo = move.promotion
    return {
        "is_promotion": promo is not None,
        "promotion_piece": chess.piece_symbol(promo).upper() if promo else None,
    }


def feat_capture(board_before: chess.Board, move: chess.Move, board_after: chess.Board) -> dict:
    return {"is_capture": board_before.is_capture(move)}


def feat_mobility(board_before: chess.Board, move: chess.Move, board_after: chess.Board) -> dict:
    """Number of legal moves available to the side to move, before and after."""
    return {
        "legal_moves_before": board_before.legal_moves.count(),
        "legal_moves_after": board_after.legal_moves.count(),
    }


# Registry of all feature functions. Add new ones here.
FEATURE_FUNCTIONS: list[Callable] = [
    feat_material,
    feat_bishop_pair,
    feat_castling_rights,
    feat_check,
    feat_en_passant,
    feat_promotion,
    feat_capture,
    feat_mobility,
]


def compute_features(board_before: chess.Board, move: chess.Move, board_after: chess.Board) -> dict:
    """Run every registered feature function and merge results into one dict."""
    row = {}
    for fn in FEATURE_FUNCTIONS:
        row.update(fn(board_before, move, board_after))
    return row


# ---------------------------------------------------------------------------
# Game / file level iteration
# ---------------------------------------------------------------------------

@dataclass
class GameMeta:
    white_elo: Optional[int] = None
    black_elo: Optional[int] = None
    result: Optional[str] = None
    time_control: Optional[str] = None
    termination: Optional[str] = None


def _parse_game_meta(game: chess.pgn.Game) -> GameMeta:
    headers = game.headers
    def _int_or_none(x):
        try:
            return int(x)
        except (ValueError, TypeError):
            return None
    return GameMeta(
        white_elo=_int_or_none(headers.get("WhiteElo")),
        black_elo=_int_or_none(headers.get("BlackElo")),
        result=headers.get("Result"),
        time_control=headers.get("TimeControl"),
        termination=headers.get("Termination"),
    )


def iter_game_rows(game: chess.pgn.Game):
    """Yield one feature-row dict per ply (half-move) in a parsed PGN game."""
    meta = _parse_game_meta(game)
    board = game.board()
    ply = 0
    for move in game.mainline_moves():
        ply += 1
        side_to_move_white = board.turn == chess.WHITE
        board_before = board.copy()
        board.push(move)
        row = compute_features(board_before, move, board)
        row.update({
            "ply": ply,
            "move_number": (ply + 1) // 2,
            "side_to_move_was_white": side_to_move_white,
            "player_elo": meta.white_elo if side_to_move_white else meta.black_elo,
            "opponent_elo": meta.black_elo if side_to_move_white else meta.white_elo,
            "white_elo": meta.white_elo,
            "black_elo": meta.black_elo,
            "result": meta.result,
            "time_control": meta.time_control,
            "termination": meta.termination,
            "san": board_before.san(move),
            "uci": move.uci(),
        })
        yield row


def process_pgn_file(
    path: str,
    max_games: Optional[int] = None,
    min_elo: Optional[int] = None,
) -> pd.DataFrame:
    """
    Parse a (plain, not zstd-compressed) .pgn file and return a DataFrame
    with one row per ply, features + metadata included.

    For large Lichess monthly dumps (zstd-compressed, many GB), use
    process_pgn_zst_stream instead so you never load the whole file at once.
    """
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        games_read = 0
        pbar = tqdm(desc="games", total=max_games)
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            meta = _parse_game_meta(game)
            if min_elo is not None:
                we, be = meta.white_elo, meta.black_elo
                if we is None or be is None or min(we, be) < min_elo:
                    continue
            rows.extend(iter_game_rows(game))
            games_read += 1
            pbar.update(1)
            if max_games is not None and games_read >= max_games:
                break
        pbar.close()
    return pd.DataFrame(rows)


def process_pgn_zst_stream(
    path: str,
    max_games: Optional[int] = 5000,
    min_elo: Optional[int] = None,
) -> pd.DataFrame:
    """
    Stream-decompress a .pgn.zst file (the format Lichess database dumps
    ship in: https://database.lichess.org/) without ever holding the whole
    file in memory. Good for pulling e.g. the first N games matching a
    rating filter out of a multi-GB monthly dump.
    """
    import zstandard as zstd

    rows = []
    games_read = 0
    with open(path, "rb") as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
            pbar = tqdm(desc="games", total=max_games)
            while True:
                game = chess.pgn.read_game(text_stream)
                if game is None:
                    break
                meta = _parse_game_meta(game)
                if min_elo is not None:
                    we, be = meta.white_elo, meta.black_elo
                    if we is None or be is None or min(we, be) < min_elo:
                        continue
                rows.extend(iter_game_rows(game))
                games_read += 1
                pbar.update(1)
                if max_games is not None and games_read >= max_games:
                    break
            pbar.close()
    return pd.DataFrame(rows)
