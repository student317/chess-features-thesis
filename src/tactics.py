"""
tactics.py

Rule-based detectors for the tactical situations of the "Step method"
books. Every detector answers ONE question:

    "Does playing `move` in `board` realise tactic X?"

so the same detector can be used for
  * a MISSED tactic:   detect(position, best_move)
  * an ALLOWED tactic: detect(position after the player's mistake,
                              the opponent's best reply)

Each detector returns a `Detection` (what was found, which squares are
involved - used for drawing arrows) or None. All definitions are written
in the docstrings, and `notebooks/03_tactics.ipynb` shows every rule on
hand-made positions (expected result vs. what the code says).

Conventions
-----------
* `board` is the position BEFORE `move`; `move` must be legal there.
* Piece values: P=1, N=3, B=3, R=5, Q=9, K=100 (king is never "won").
* "Defenders / attackers of a square" use `board.attackers(color, sq)`.
  Pinned pieces still count as attackers/defenders (a simplification).
* The material outcome of captures is judged with a static exchange
  evaluation (SEE): both sides keep capturing on the square with their
  cheapest attacker and may stop whenever continuing loses material.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import chess

VALUE = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 100,
}

MATE_SCORE = 10000     # same constant as in critical.py (cp used for mates)
MATE_MARGIN = 200      # scores within this distance of +-MATE_SCORE are mates

# tactic ids
CAPTURE_UNPROTECTED = "capture_unprotected"
DEFEND_THREATENED = "defend_threatened_piece"
FAVOURABLE_EXCHANGE = "favourable_exchange"
CAPTURE_ATTACKED_MORE = "capture_attacked_more_than_defended"
DOUBLE_ATTACK = "double_attack"
PIN = "pin"
ELIMINATION = "elimination_of_defence"
FORCED_MATE = "forced_mate"
DISCOVERED_ATTACK = "discovered_attack"
DEFEND_AGAINST_MATE = "defend_against_mate"


@dataclass
class Detection:
    tactic: str
    note: str = ""
    details: dict = field(default_factory=dict)
    arrows: list = field(default_factory=list)      # [(from_square, to_square)]


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def _see_square(board: chess.Board, square: int, side: bool) -> int:
    """Material that `side` can win by capturing the piece standing on
    `square`, assuming both sides then keep capturing there with their
    cheapest attacker and each side may stop. Never negative (side can
    decline to capture). Kings cannot capture a defended piece."""
    victim = board.piece_type_at(square)
    if victim is None or victim == chess.KING:
        return 0
    victim_value = VALUE[victim]
    attackers = sorted(board.attackers(side, square), key=lambda s: VALUE[board.piece_type_at(s)])
    for a in attackers:
        b2 = board.copy(stack=False)
        piece = b2.remove_piece_at(a)
        b2.set_piece_at(square, piece)
        if piece.piece_type == chess.KING and b2.attackers(not side, square):
            continue                                   # king may not capture into a defended square
        return max(0, victim_value - _see_square(b2, square, not side))
    return 0


def see(board: chess.Board, move: chess.Move) -> int:
    """Net material (in pawns) the mover gains by playing the capture
    `move` if the exchange on that square is played out. Can be negative
    (the capture loses material). Non-captures give 0 - 'cost of recapture'."""
    mover = board.piece_at(move.from_square)
    if mover is None:
        return 0
    b2 = board.copy(stack=False)
    victim_value = 0
    if board.is_en_passant(move):
        victim_value = VALUE[chess.PAWN]
        b2.remove_piece_at(chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square)))
    else:
        vt = board.piece_type_at(move.to_square)
        if vt is not None:
            victim_value = VALUE[vt]
    piece = b2.remove_piece_at(move.from_square)
    if move.promotion:
        piece = chess.Piece(move.promotion, piece.color)
    b2.set_piece_at(move.to_square, piece)
    return victim_value - _see_square(b2, move.to_square, not piece.color)


def _remains_protected(board: chess.Board, from_sq: int, to_sq: int) -> bool:
    """After the piece went from_sq -> to_sq, does another piece of its
    colour still defend to_sq?"""
    b2 = board.copy(stack=False)
    piece = b2.remove_piece_at(from_sq)
    b2.set_piece_at(to_sq, piece)
    return len(b2.attackers(piece.color, to_sq)) > 0


def capture_labels(board: chess.Board, from_sq: int, to_sq: int, min_victim_value: int = 1) -> set:
    """Which 'good capture' tactics apply to the (hypothetical) capture
    from_sq x to_sq? The turn is ignored, so this can also be asked about
    a capture the mover would make LATER (used by fork / discovered attack).

      capture_unprotected      nobody of the victim's side attacks to_sq
      favourable_exchange      victim worth MORE than the capturer, victim is
                               defended (so the capturer can be recaptured),
                               and SEE > 0
      capture_attacked_more_than_defended
                               victim worth <= the capturer, my attackers of
                               to_sq > their defenders, SEE > 0, and the
                               capturer is still protected afterwards
    """
    mover = board.piece_at(from_sq)
    victim = board.piece_at(to_sq)
    if mover is None or victim is None or mover.color == victim.color or victim.piece_type == chess.KING:
        return set()
    mv, vv = VALUE[mover.piece_type], VALUE[victim.piece_type]
    if vv < min_victim_value:
        return set()
    me, opp = mover.color, victim.color
    n_def = len(board.attackers(opp, to_sq))
    n_att = len(board.attackers(me, to_sq))
    gain = see(board, chess.Move(from_sq, to_sq))
    labels = set()
    if n_def == 0:
        labels.add(CAPTURE_UNPROTECTED)
    elif vv > mv and gain > 0:
        labels.add(FAVOURABLE_EXCHANGE)
    elif vv <= mv and n_att > n_def and gain > 0 and _remains_protected(board, from_sq, to_sq):
        labels.add(CAPTURE_ATTACKED_MORE)
    return labels


def _is_safe_piece(board: chess.Board, square: int) -> bool:
    """True if the opponent cannot win or trade off the piece on `square`:
    it is not attacked, or it is defended and every attacker is worth more
    than the piece itself."""
    piece = board.piece_at(square)
    attackers = board.attackers(not piece.color, square)
    if not attackers:
        return True
    if not board.attackers(piece.color, square):
        return False
    cheapest = min(VALUE[board.piece_type_at(a)] for a in attackers)
    return cheapest > VALUE[piece.piece_type]


def _threatened(board: chess.Board, color: bool, min_value: int = 1) -> set:
    """Squares of `color`'s pieces (not the king) that the opponent could
    capture and win material (SEE > 0)."""
    out = set()
    for sq, piece in board.piece_map().items():
        if piece.color == color and piece.piece_type != chess.KING and VALUE[piece.piece_type] >= min_value:
            if _see_square(board, sq, not color) > 0:
                out.add(sq)
    return out


def _has_winning_capture(board: chess.Board, from_sq: int) -> Optional[tuple]:
    """First opponent piece the piece on from_sq could capture with one of
    the 'good capture' labels (turn ignored), as (square, labels)."""
    mover = board.piece_at(from_sq)
    for t in board.attacks(from_sq):
        p = board.piece_at(t)
        if p is not None and p.color != mover.color and p.piece_type != chess.KING:
            labels = capture_labels(board, from_sq, t)
            if labels:
                return t, labels
    return None


# ---------------------------------------------------------------------------
# Detectors  (all accept **ctx and ignore keys they do not need)
# ---------------------------------------------------------------------------

def detect_capture_unprotected(board, move, **ctx) -> Optional[Detection]:
    """1. Capturing an unprotected piece.
    The move is a capture of a non-king piece that no opposing piece
    attacks (defends), so the capture wins it for nothing."""
    if CAPTURE_UNPROTECTED in capture_labels(board, move.from_square, move.to_square):
        return Detection(CAPTURE_UNPROTECTED, "captures an undefended piece",
                         arrows=[(move.from_square, move.to_square)])
    return None


def detect_favourable_exchange(board, move, **ctx) -> Optional[Detection]:
    """3. Favourable exchange.
    The move captures a MORE valuable, defended piece; the capturer can be
    recaptured, but the exchange still wins material (SEE > 0)."""
    if FAVOURABLE_EXCHANGE in capture_labels(board, move.from_square, move.to_square):
        return Detection(FAVOURABLE_EXCHANGE, f"wins material in the exchange (SEE = {see(board, move)})",
                         {"see": see(board, move)}, [(move.from_square, move.to_square)])
    return None


def detect_capture_attacked_more(board, move, **ctx) -> Optional[Detection]:
    """4. Capturing a piece that is attacked more often than defended.
    The move captures a piece worth <= the capturer, our attackers of the
    square outnumber their defenders, SEE > 0 and the capturer stays
    protected."""
    if CAPTURE_ATTACKED_MORE in capture_labels(board, move.from_square, move.to_square):
        return Detection(CAPTURE_ATTACKED_MORE, f"wins material (SEE = {see(board, move)})",
                         {"see": see(board, move)}, [(move.from_square, move.to_square)])
    return None


def detect_defend_threatened_piece(board, move, **ctx) -> Optional[Detection]:
    """2. Defending a piece that would otherwise be captured.
    Before the move one of our pieces X is threatened (the opponent can
    capture it and win material). The move is not a capture, X stays where
    it is, afterwards X is no longer threatened, and the moved piece now
    defends X (it did not before)."""
    mover = board.piece_at(move.from_square)
    me, opp = mover.color, not mover.color
    if board.is_capture(move):
        return None
    before = _threatened(board, me)
    if not before:
        return None
    after = board.copy(stack=False)
    after.push(move)
    for s in sorted(before):
        if s == move.from_square:
            continue                                   # the piece itself moved away: not a defence
        if _see_square(after, s, opp) > 0:
            continue                                   # still threatened
        if move.to_square in after.attackers(me, s) and move.from_square not in board.attackers(me, s):
            return Detection(DEFEND_THREATENED, f"protects the threatened piece on {chess.square_name(s)}",
                             {"protected_square": chess.square_name(s)}, [(move.to_square, s)])
    return None


def detect_double_attack(board, move, min_victim: int = 1, **ctx) -> Optional[Detection]:
    """5. Double attack (fork).
    After the move the moved piece (not the king) attacks at least two
    'targets', where a target is either the enemy king (the move gives
    check with this piece) or an enemy piece it could capture with a good
    capture (labels of capture_labels). The forking piece must be safe
    (not attacked, or defended and attacked only by more valuable pieces).
    'Important square' targets are NOT implemented."""
    mover = board.piece_at(move.from_square)
    if mover.piece_type == chess.KING or board.is_castling(move):
        return None
    me, opp = mover.color, not mover.color
    after = board.copy(stack=False)
    after.push(move)
    to = move.to_square
    if not _is_safe_piece(after, to):
        return None
    targets = []
    if after.is_check() and to in after.checkers():
        targets.append(("king", after.king(opp), []))
    for t in after.attacks(to):
        p = after.piece_at(t)
        if p is not None and p.color == opp and p.piece_type != chess.KING:
            labels = capture_labels(after, to, t, min_victim)
            if labels:
                targets.append((chess.piece_name(p.piece_type), t, sorted(labels)))
    if len(targets) >= 2:
        names = ", ".join(f"{n} {chess.square_name(t)}" for n, t, _ in targets)
        return Detection(DOUBLE_ATTACK, f"attacks {names}",
                         {"targets": [(n, chess.square_name(t), l) for n, t, l in targets]},
                         [(to, t) for _, t, _ in targets])
    return None


_DIAG = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
_ORTH = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _directions(piece_type: int) -> list:
    if piece_type == chess.BISHOP:
        return _DIAG
    if piece_type == chess.ROOK:
        return _ORTH
    if piece_type == chess.QUEEN:
        return _DIAG + _ORTH
    return []


def _first_two_pieces(board: chess.Board, start: int, df: int, dr: int) -> list:
    out = []
    f, r = chess.square_file(start), chess.square_rank(start)
    while len(out) < 2:
        f += df
        r += dr
        if not (0 <= f < 8 and 0 <= r < 8):
            break
        sq = chess.square(f, r)
        if board.piece_at(sq) is not None:
            out.append(sq)
    return out


def _on_ray(start: int, df: int, dr: int, target: int) -> bool:
    f, r = chess.square_file(start), chess.square_rank(start)
    while True:
        f += df
        r += dr
        if not (0 <= f < 8 and 0 <= r < 8):
            return False
        if chess.square(f, r) == target:
            return True


def _pins_from(board: chess.Board, sq: int) -> list:
    """Pins made by the bishop/rook/queen on `sq` (on `board`, safety of the
    pinner NOT checked): [(front_sq, back_sq, kind, df, dr)]. A pin is an enemy
    piece F (not king) with an enemy piece B directly behind it on a line of
    the pinner; B is the king (absolute) or worth MORE than F and would be won
    (SEE > 0) once F has left the line (relative)."""
    piece = board.piece_at(sq)
    opp = not piece.color
    out = []
    for df, dr in _directions(piece.piece_type):
        occ = _first_two_pieces(board, sq, df, dr)
        if len(occ) < 2:
            continue
        front_sq, back_sq = occ
        front, back = board.piece_at(front_sq), board.piece_at(back_sq)
        if front.color != opp or back.color != opp or front.piece_type == chess.KING:
            continue
        if back.piece_type == chess.KING:
            kind = "absolute"
        elif VALUE[back.piece_type] > VALUE[front.piece_type]:
            hypo = board.copy(stack=False)
            hypo.remove_piece_at(front_sq)
            if see(hypo, chess.Move(sq, back_sq)) <= 0:
                continue
            kind = "relative"
        else:
            continue
        out.append((front_sq, back_sq, kind, df, dr))
    return out


def detect_pin(board, move, opp_reply: Optional[chess.Move] = None, **ctx) -> Optional[Detection]:
    """6. Pin (a NEW pin).
    After the move the moved piece is a safe bishop, rook or queen that pins
    an enemy piece F to a piece B behind it (see _pins_from: B is the king, or
    worth more than F and winnable once F leaves the line). The pair (F, B)
    must NOT already have been pinned by one of our pieces before the move -
    a move that merely keeps an existing pin (e.g. the pinning queen retreats
    along the same line) is not a pin tactic.
    If `opp_reply` (the opponent's best answer) is given, the pin only counts
    when that reply does not move F off the line.
    Not implemented: pins to an 'important square', skewers."""
    if board.is_castling(move):
        return None
    after = board.copy(stack=False)
    after.push(move)
    to = move.to_square
    piece = after.piece_at(to)
    if piece is None or piece.piece_type not in (chess.BISHOP, chess.ROOK, chess.QUEEN):
        return None
    if not _is_safe_piece(after, to):
        return None
    me = piece.color
    existing = set()
    for sq, p in board.piece_map().items():
        if p.color == me and p.piece_type in (chess.BISHOP, chess.ROOK, chess.QUEEN):
            existing |= {(f, b) for f, b, *_ in _pins_from(board, sq)}
    best = None
    for front_sq, back_sq, kind, df, dr in _pins_from(after, to):
        if (front_sq, back_sq) in existing:
            continue
        if opp_reply is not None and opp_reply.from_square == front_sq and not _on_ray(to, df, dr, opp_reply.to_square):
            continue                                   # the pinned piece simply walks away
        score = VALUE[after.piece_at(back_sq).piece_type]
        if best is None or score > best[0]:
            best = (score, kind, front_sq, back_sq)
    if best is None:
        return None
    _, kind, front_sq, back_sq = best
    return Detection(PIN, f"{kind} pin: {chess.square_name(front_sq)} is pinned to {chess.square_name(back_sq)}",
                     {"kind": kind, "pinned": chess.square_name(front_sq), "behind": chess.square_name(back_sq)},
                     [(to, front_sq), (front_sq, back_sq)])


# --- building blocks shared by tactics: "what does this piece threaten?" -----
# A THREAT of a piece of ours (position AFTER our move) is one of
#   king      the piece gives check
#   material  it attacks an enemy piece it could capture with a good capture
#   square    it attacks an EMPTY square X such that, if it went there (our
#             turn), it would give checkmate ("mating square") or fork two
#             targets ("fork square"). This is the "important square".
# Layering (no infinite recursion): king/material are plain facts,
# a square is defined through them one level deeper (mate / >=2 targets).
_RANK = {"king": 0, "material": 1, "square": 2, "none": 3}


def _piece_targets(board: chess.Board, sq: int, me: bool, min_victim: int = 1) -> list:
    """King and material targets of our piece on `sq`; `board` is the
    position after our move (opponent to move). [(kind, square, labels)]"""
    opp = not me
    out = []
    if board.turn == opp and sq in board.checkers():
        out.append(("king", board.king(opp), ""))
    for t in board.attacks(sq):
        p = board.piece_at(t)
        if p is not None and p.color == opp and p.piece_type != chess.KING:
            labels = capture_labels(board, sq, t, min_victim)
            if labels:
                out.append(("material", t, ",".join(sorted(labels))))
    return out


def _square_targets(board: chess.Board, sq: int, me: bool, only: Optional[set] = None, min_victim: int = 1) -> list:
    """Important-square targets of our piece on `sq` (see above). `only`
    restricts the squares (e.g. the newly attacked ones). Skipped when the
    opponent is in check (the opponent must answer the check anyway)."""
    if board.is_check():
        return []
    hyp = board.copy(stack=False)
    hyp.turn = me
    hyp.ep_square = None
    out = []
    for x in board.attacks(sq):
        if (only is not None and x not in only) or board.piece_at(x) is not None:
            continue
        mv = chess.Move(sq, x)
        if mv not in hyp.legal_moves:
            continue
        h2 = hyp.copy(stack=False)
        h2.push(mv)
        if h2.is_checkmate():
            out.append(("square", x, "mating square"))
        elif _is_safe_piece(h2, x) and len(_piece_targets(h2, x, me, min_victim)) >= 2:
            out.append(("square", x, "fork square"))
    return out


def _top(threats: list) -> str:
    return min((t[0] for t in threats), key=_RANK.get) if threats else "none"


def detect_discovered_attack(board, move, min_victim: int = 1, **ctx) -> Optional[Detection]:
    """9. Discovered attack.
    A bishop/rook/queen of ours (the back piece B, not the moved piece M)
    attacks, after the move, a square it did not attack before - M cleared
    its line. B must be safe, unless it gives check (the opponent must answer). The newly attacked squares must hold a
    THREAT of B (king = discovered check, material = a good capture,
    square = an important square, see _piece_targets / _square_targets).
    Classification `kind` = the strongest threat of B + the strongest threat
    of the moved piece M, ordered king < material < square:
        king+king, king+material, king+square, material+material,
        material+square, square+square      (the book's combinations)
        king+none, material+none, square+none   (only B threatens something)"""
    mover = board.piece_at(move.from_square)
    if board.is_castling(move):
        return None
    me = mover.color
    after = board.copy(stack=False)
    after.push(move)
    to = move.to_square
    m_threats = None                                   # computed lazily, only if needed
    best = None
    for s, p in board.piece_map().items():
        if p.color != me or s == move.from_square or p.piece_type not in (chess.BISHOP, chess.ROOK, chess.QUEEN):
            continue
        new_squares = set(after.attacks(s)) - set(board.attacks(s))
        if not new_squares:
            continue
        b_threats = [t for t in _piece_targets(after, s, me, min_victim) if t[1] in new_squares]
        if not any(t[0] == "king" for t in b_threats):
            # without a check the opponent may simply take B, so B must be safe
            if not _is_safe_piece(after, s):
                continue
            b_threats += _square_targets(after, s, me, only=new_squares, min_victim=min_victim)
        if not b_threats:
            continue
        if m_threats is None:
            m_threats = _piece_targets(after, to, me, min_victim) + _square_targets(after, to, me, min_victim=min_victim)
        b_top, m_top = _top(b_threats), _top(m_threats)
        kind = "+".join(sorted((b_top, m_top), key=_RANK.get))
        key = tuple(sorted((_RANK[b_top], _RANK[m_top])))
        if best is None or key < best[0]:
            best = (key, kind, s, b_threats, m_threats)
    if best is None:
        return None
    _, kind, s, b_threats, m_threats = best
    bt = min(b_threats, key=lambda t: _RANK[t[0]])
    arrows = [(s, bt[1])]
    note = f"{kind}: {chess.square_name(s)} now threatens {bt[0]} {chess.square_name(bt[1])}"
    if m_threats:
        mt = min(m_threats, key=lambda t: _RANK[t[0]])
        arrows.append((to, mt[1]))
        note += f"; moved piece threatens {mt[0]} {chess.square_name(mt[1])}"
    return Detection(DISCOVERED_ATTACK, note,
                     {"kind": kind, "back_piece": chess.square_name(s),
                      "back_threats": [(k, chess.square_name(q), l) for k, q, l in b_threats],
                      "moved_threats": [(k, chess.square_name(q), l) for k, q, l in m_threats or []]},
                     arrows)


def detect_elimination_of_defence(board, move, pv: Optional[list] = None, **ctx) -> Optional[Detection]:
    """7. Elimination of the defence (needs the engine's line, `pv` =
    [our move, opponent reply, our second move], pv[0] == move).
    The second move captures an enemy piece T with a good capture; T was
    NOT a good capture before the first move (so it was adequately
    defended), and after the first two moves T has fewer defenders.
    Sub-kinds, in this order:
      capture_defender  the first move captures a defender of T
      lure              the opponent's reply is a defender of T capturing our
                        first piece (sacrifice / exchange)
      chase             the first move attacks a defender of T and the reply
                        moves it away
      other             defenders were reduced in another way"""
    if not pv or len(pv) < 3 or pv[0] != move:
        return None
    m1, r1, m2 = pv[0], pv[1], pv[2]
    me = board.color_at(m1.from_square)
    if me is None or m1 not in board.legal_moves:
        return None
    opp = not me
    b1 = board.copy(stack=False)
    b1.push(m1)
    if r1 not in b1.legal_moves:
        return None
    b2 = b1.copy(stack=False)
    b2.push(r1)
    if m2 not in b2.legal_moves:
        return None
    t = m2.to_square
    target_after, target_before = b2.piece_at(t), board.piece_at(t)
    if target_after is None or target_before is None or target_after.color != opp:
        return None
    if target_before.piece_type != target_after.piece_type or target_before.color != opp:
        return None
    if not capture_labels(b2, m2.from_square, t):
        return None
    if any(capture_labels(board, a, t) for a in board.attackers(me, t)):
        return None                                    # it was already a winning capture
    defenders_before = set(board.attackers(opp, t))
    defenders_after = set(b2.attackers(opp, t))
    if len(defenders_after) >= len(defenders_before):
        return None
    if m1.to_square in defenders_before:
        kind = "capture_defender"
    elif r1.from_square in defenders_before and r1.to_square == m1.to_square:
        kind = "lure"
    elif r1.from_square in defenders_before and r1.from_square in b1.attacks(m1.to_square):
        kind = "chase"
    else:
        kind = "other"
    return Detection(ELIMINATION, f"{kind}: {chess.square_name(t)} loses its defender",
                     {"kind": kind, "target": chess.square_name(t)},
                     [(m1.from_square, m1.to_square), (m2.from_square, t)])


def mate_in_from_cp(cp: Optional[int]) -> Optional[int]:
    """Moves until WE mate, from an engine score (cp, our point of view,
    mates stored as +-(MATE_SCORE - n)). None if it is not a mate score."""
    if cp is not None and cp >= MATE_SCORE - MATE_MARGIN:
        return MATE_SCORE - cp
    return None


def mated_in_from_cp(cp: Optional[int]) -> Optional[int]:
    """Moves until WE are mated (None if the score is not a mate against us)."""
    if cp is not None and cp <= -(MATE_SCORE - MATE_MARGIN):
        return MATE_SCORE + cp
    return None


def detect_forced_mate(board, move, best_cp: Optional[int] = None, max_mate: int = 5, **ctx) -> Optional[Detection]:
    """8. Forced mate.
    With `best_cp` (engine score of the BEST move, our point of view): a
    mate in at most `max_mate` moves. Without it: the move itself gives
    checkmate (mate in 1, checked on the board)."""
    if best_cp is not None:
        n = mate_in_from_cp(best_cp)
        if n is not None and 1 <= n <= max_mate:
            return Detection(FORCED_MATE, f"engine: forced mate in {n}", {"mate_in": n},
                             [(move.from_square, move.to_square)])
        return None
    after = board.copy(stack=False)
    after.push(move)
    if after.is_checkmate():
        return Detection(FORCED_MATE, "the move gives checkmate", {"mate_in": 1},
                         [(move.from_square, move.to_square)])
    return None


def _side_to_move_mates_in_one(board: chess.Board) -> bool:
    b = board.copy(stack=False)
    for mv in list(b.legal_moves):
        b.push(mv)
        mate = b.is_checkmate()
        b.pop()
        if mate:
            return True
    return False


def detect_defend_against_mate(board, move, best_cp: Optional[int] = None, second_cp: Optional[int] = None,
                               max_mate: int = 5, **ctx) -> Optional[Detection]:
    """10. Defending against mate.
    Every legal move except `move` allows a forced mate.
    With engine scores (`best_cp`, `second_cp` of the best and second-best
    move): the second-best move is mated in <= max_mate moves while the
    best move is not (so every alternative loses to mate too).
    Without scores: the same idea for mate in ONE - `move` is the only
    legal move after which the opponent has no mate in one."""
    if best_cp is not None and second_cp is not None:
        n = mated_in_from_cp(second_cp)
        if n is not None and 1 <= n <= max_mate and mated_in_from_cp(best_cp) is None:
            return Detection(DEFEND_AGAINST_MATE, f"all other moves allow mate (in {n})", {"mate_in": n},
                             [(move.from_square, move.to_square)])
        return None
    legal = list(board.legal_moves)
    if len(legal) < 2 or move not in legal:
        return None
    safe = []
    for mv in legal:
        b2 = board.copy(stack=False)
        b2.push(mv)
        if not _side_to_move_mates_in_one(b2):
            safe.append(mv)
    if safe == [move]:
        return Detection(DEFEND_AGAINST_MATE, "the only move that stops mate in one", {"mate_in": 1},
                         [(move.from_square, move.to_square)])
    return None


# ---------------------------------------------------------------------------
# Registry / convenience
# ---------------------------------------------------------------------------

DETECTORS: dict = {
    CAPTURE_UNPROTECTED: detect_capture_unprotected,
    DEFEND_THREATENED: detect_defend_threatened_piece,
    FAVOURABLE_EXCHANGE: detect_favourable_exchange,
    CAPTURE_ATTACKED_MORE: detect_capture_attacked_more,
    DOUBLE_ATTACK: detect_double_attack,
    PIN: detect_pin,
    ELIMINATION: detect_elimination_of_defence,
    FORCED_MATE: detect_forced_mate,
    DISCOVERED_ATTACK: detect_discovered_attack,
    DEFEND_AGAINST_MATE: detect_defend_against_mate,
}


def detect_all(board: chess.Board, move: chess.Move, only: Optional[list] = None, **ctx) -> list:
    """Run every detector (or only those named in `only`). ctx may hold:
    pv, opp_reply, best_cp, second_cp."""
    found = []
    for name, fn in DETECTORS.items():
        if only is not None and name not in only:
            continue
        d = fn(board, move, **ctx)
        if d is not None:
            found.append(d)
    return found


def classify_mistake(board: chess.Board, best_move: chess.Move, played_move: chess.Move,
                     opp_reply: Optional[chess.Move] = None, **ctx) -> dict:
    """For a mistake (played_move != best_move):
      missed  - tactics the best move would have realised
      allowed - tactics the opponent realises with `opp_reply` (its best
                answer to the played move; None = not evaluated)"""
    missed = detect_all(board, best_move, **ctx)
    allowed = []
    if opp_reply is not None:
        after = board.copy(stack=False)
        after.push(played_move)
        allowed = detect_all(after, opp_reply)
    return {"missed": missed, "allowed": allowed}
