"""
visualize.py

Render a parsed PGN game as a sequence of board diagrams (SVG), for use
inside a Jupyter notebook - e.g. to sanity-check that a game's moves do
what you expect, or to show a specific game when writing up the thesis.
"""

from __future__ import annotations

from typing import Iterator

import chess
import chess.pgn
import chess.svg
from IPython.display import HTML, display


def load_games(path: str) -> list[chess.pgn.Game]:
    """Read every game from a plain (uncompressed) PGN file into memory."""
    games = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            games.append(game)
    return games


def game_board_states(game: chess.pgn.Game) -> Iterator[tuple[int, str, chess.Move, chess.Board]]:
    """Yield (ply, san, move, board_after_copy) for every move in a game."""
    board = game.board()
    ply = 0
    for move in game.mainline_moves():
        ply += 1
        san = board.san(move)
        board.push(move)
        yield ply, san, move, board.copy()


def show_game(game: chess.pgn.Game, size: int = 220, per_row: int = 4) -> None:
    """
    Render every position in `game` as a small SVG board, in a grid, with
    the move that led to each position as a caption and that move's
    squares highlighted. Call this from a notebook cell - it displays
    directly, it doesn't return anything to print.
    """
    headers = game.headers
    title = f"{headers.get('White', '?')} vs {headers.get('Black', '?')} — {headers.get('Event', '')}"

    cells = []
    start_board = game.board()
    start_svg = chess.svg.board(start_board, size=size)
    cells.append(f"<div style='text-align:center;margin:4px'>{start_svg}<div><b>start</b></div></div>")

    for ply, san, move, board_after in game_board_states(game):
        svg = chess.svg.board(board_after, lastmove=move, size=size)
        move_number = (ply + 1) // 2
        label = f"{move_number}{'.' if ply % 2 == 1 else '...'} {san}"
        cells.append(f"<div style='text-align:center;margin:4px'>{svg}<div><b>{label}</b></div></div>")

    rows = []
    for i in range(0, len(cells), per_row):
        row_html = "".join(cells[i:i + per_row])
        rows.append(f"<div style='display:flex;flex-wrap:wrap'>{row_html}</div>")

    display(HTML(f"<h4>{title}</h4>" + "".join(rows)))
