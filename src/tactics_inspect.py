"""
tactics_inspect.py

Tools to LOOK at what the detectors in tactics.py do:

  run_cases(cases)       hand-made positions with the expected answer ->
                         PASS/FAIL table + a board with arrows for each case
  check_see(cases)       expected values of the static exchange evaluation
  inspect_move(fen, uci) every detector on ANY position (e.g. a row of
                         critical_positions.csv) with arrows

Arrow colours: blue = the move, red = opponent reply (pv), green = what the
detector found (or the 2nd move of a pv).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import chess
import chess.svg
import pandas as pd
from IPython.display import HTML, display

import tactics as T

BLUE, RED, GREEN = "#2060e0cc", "#d03030cc", "#20a040cc"


@dataclass
class Case:
    name: str
    tactic: str
    fen: str
    uci: str
    expect: bool
    pv: Optional[list] = None            # list of UCI moves, pv[0] == uci
    ctx: dict = field(default_factory=dict)
    note: str = ""
    expect_kind: str = ""                # optional: expected details["kind"] of the detection


def _arrows(move, detection=None, pv=None):
    arrows = []
    if pv:
        for m, col in zip(pv, (BLUE, RED, GREEN)):
            arrows.append(chess.svg.Arrow(m.from_square, m.to_square, color=col))
    else:
        arrows.append(chess.svg.Arrow(move.from_square, move.to_square, color=BLUE))
        if detection is not None:
            for a, b in detection.arrows:
                if (a, b) != (move.from_square, move.to_square):
                    arrows.append(chess.svg.Arrow(a, b, color=GREEN))
    return arrows


def _cell(board, arrows, caption, color, size):
    svg = chess.svg.board(board, arrows=arrows, size=size)
    return (f"<div style='margin:6px;text-align:center;max-width:{size + 20}px'>{svg}"
            f"<div style='font-size:12px;color:{color}'>{caption}</div></div>")


def run_cases(cases: list, show: bool = True, size: int = 240) -> pd.DataFrame:
    rows, cells = [], []
    for c in cases:
        board = chess.Board(c.fen)
        move = chess.Move.from_uci(c.uci)
        if move not in board.legal_moves:
            rows.append({"case": c.name, "tactic": c.tactic, "expected": c.expect,
                         "detected": None, "result": "BAD CASE (illegal move)", "details": "", "also detected": ""})
            continue
        ctx = dict(c.ctx)
        pv = [chess.Move.from_uci(u) for u in c.pv] if c.pv else None
        if pv:
            ctx["pv"] = pv
        det = T.DETECTORS[c.tactic](board, move, **ctx)
        ok = (det is not None) == c.expect
        if ok and det is not None and c.expect_kind:
            ok = det.details.get("kind") == c.expect_kind
        others = [d.tactic for d in T.detect_all(board, move, **ctx) if d.tactic != c.tactic]
        rows.append({"case": c.name, "tactic": c.tactic, "expected": c.expect, "detected": det is not None,
                     "result": "PASS" if ok else "FAIL", "details": det.note if det else "",
                     "also detected": ", ".join(others)})
        caption = (f"<b>{c.name}</b>: expected {'YES' if c.expect else 'NO'}, "
                   f"detected {'YES' if det else 'NO'} - <b>{'PASS' if ok else 'FAIL'}</b>"
                   + (f"<br>{det.note}" if det else "") + (f"<br>expected kind: {c.expect_kind}" if c.expect_kind else "") + (f"<br><i>{c.note}</i>" if c.note else "")
                   + ("<br><i>SCORE TYPED IN BY HAND - the board is NOT analysed, "
                      "it is only a placeholder</i>" if ("best_cp" in ctx or "second_cp" in ctx) else ""))
        cells.append(_cell(board, _arrows(move, det, pv), caption, "#2a7a2a" if ok else "#c02020", size))
    df = pd.DataFrame(rows)
    n_ok = int((df["result"] == "PASS").sum())
    print(f"{n_ok}/{len(df)} cases passed")
    if show and cells:
        display(HTML("<div style='display:flex;flex-wrap:wrap'>" + "".join(cells) + "</div>"))
    return df


def check_see(cases: list) -> pd.DataFrame:
    """cases: (name, fen, uci, expected_see)"""
    rows = []
    for name, fen, uci, expected in cases:
        board = chess.Board(fen)
        move = chess.Move.from_uci(uci)
        got = T.see(board, move) if move in board.legal_moves else None
        rows.append({"case": name, "move": uci, "expected SEE": expected, "SEE": got,
                     "result": "PASS" if got == expected else "FAIL"})
    df = pd.DataFrame(rows)
    print(f"{int((df['result'] == 'PASS').sum())}/{len(df)} SEE cases passed")
    return df


def inspect_move(fen: str, uci: str, size: int = 300, html_only: bool = False, **ctx):
    """Run ALL detectors for `uci` on `fen` and draw the board. ctx may
    hold pv (list of UCI strings), best_cp, second_cp, opp_reply (UCI).
    html_only=True: do not display, return (found, html_of_the_board) so that
    several boards can be put side by side."""
    board = chess.Board(fen)
    move = chess.Move.from_uci(uci)
    c = dict(ctx)
    if c.get("pv"):
        c["pv"] = [chess.Move.from_uci(u) if isinstance(u, str) else u for u in c["pv"]]
    if isinstance(c.get("opp_reply"), str):
        c["opp_reply"] = chess.Move.from_uci(c["opp_reply"])
    found = T.detect_all(board, move, **c)
    arrows = [chess.svg.Arrow(move.from_square, move.to_square, color=BLUE)]
    for d in found:
        for a, b in d.arrows:
            if (a, b) != (move.from_square, move.to_square):
                arrows.append(chess.svg.Arrow(a, b, color=GREEN))
    names = "; ".join(f"<b>{d.tactic}</b> ({d.note})" for d in found) or "no tactic recognised"
    san = board.san(move) if move in board.legal_moves else uci
    cell = _cell(board, arrows, f"best move {san}: {names}", "#222222", size)
    if html_only:
        return found, cell
    display(HTML(cell))
    return found
