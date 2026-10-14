import chess
import chess.polyglot
import math
import time
from dataclasses import dataclass
from typing import Optional

try:
    from ai_eval import (
        evaluate_ai_cp,
        evaluate_ai_batch_cp,
        clear_ai_cache,
    )
except Exception:
    def evaluate_ai_cp(board, scale=600):
        return 0

    def evaluate_ai_batch_cp(boards, scale=600):
        return [0] * len(boards)

    def clear_ai_cache():
        pass



INFINITY = 10_000_000
MATE_SCORE = 1_000_000
QUIESCENCE_DEPTH = 5
DELTA_MARGIN = 150

# Futility pruning conservador: apenas em depth baixo.
FUTILITY_MARGINS = {
    1: 110,
    2: 240,
}

# Reverse Futility Pruning (RFP): corta nós não-PV em que a avaliação
# estática já está confortavelmente acima de beta.
REVERSE_FUTILITY_MARGINS = {
    1: 120,
    2: 250,
    3: 380,
}

# Late Move Pruning (LMP): em nós não-PV e pouco profundos, depois de
# vários quiet moves já terem sido tentados, os restantes lances quietos
# de baixa prioridade raramente justificam pesquisa completa.
LMP_QUIET_LIMITS = {
    1: 6,
    2: 10,
    3: 16,
}

# 2.4: pesquisa mais paciente antes de terminar cedo.
EARLY_STOP_MIN_DEPTH = 10
EARLY_STOP_STABLE_ITERS = 3

# 2.4: recapturas materiais críticas recebem 1 ply extra.
RECAPTURE_EXTENSION_MAX_DEPTH = 8
RECAPTURE_EXTENSION_SEE_MIN = -40

# 2.4: lances de raiz materialmente arriscados recebem verificação adaptativa.
ROOT_TACTICAL_VERIFY_MARGIN = 140
ROOT_VERIFY_MINOR_PAWN_PLIES = 2
ROOT_VERIFY_ROOK_MINOR_PLIES = 2
ROOT_VERIFY_QUEEN_PLIES = 3
ROOT_VERIFY_MAX_PLIES = 3

# 2.4 deep-search: estabilidade real da Principal Variation (PV).
PV_STABILITY_LENGTH = 4
TACTICAL_PV_SCAN_PLIES = 6

# Extensões internas seletivas. Só se acumula no máximo 1 ply por nó,
# para evitar explosões da árvore.
SINGULAR_EXTENSION_MIN_DEPTH = 7
SINGULAR_EXTENSION_MARGIN = 70
SELECTIVE_CHECK_EXTENSION_MAX_DEPTH = 8
SELECTIVE_CHECK_MAX_REPLIES = 2
PASSED_PAWN_EXTENSION_MAX_DEPTH = 8

# Avaliação material rápida usada como pré-filtro para pruning.
# Quando existe dúvida, a engine mantém a pesquisa em vez de cortar.
FAST_EVAL_SAFETY_MARGIN = 90
MAX_FAST_EVAL_CACHE_ENTRIES = 50_000

# History heuristics limitadas para não crescerem sem controlo.
HISTORY_MAX = 20_000
COUNTERMOVE_BONUS = 60_000

AI_SCALE = 600
AI_ROOT_ORDER_WEIGHT = 1.0
USE_AI = True

TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2

MAX_TT_ENTRIES = 200_000


PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 950,
    chess.KING: 0,
}


PHASE_VALUES = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 1,
    chess.ROOK: 2,
    chess.QUEEN: 4,
    chess.KING: 0,
}


PAWN_TABLE = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [5, 10, 10, -20, -20, 10, 10, 5],
    [5, -5, -10, 0, 0, -10, -5, 5],
    [0, 0, 0, 20, 20, 0, 0, 0],
    [5, 5, 10, 25, 25, 10, 5, 5],
    [10, 10, 20, 30, 30, 20, 10, 10],
    [50, 50, 50, 50, 50, 50, 50, 50],
    [0, 0, 0, 0, 0, 0, 0, 0],
]


KNIGHT_TABLE = [
    [-50, -40, -30, -30, -30, -30, -40, -50],
    [-40, -20, 0, 0, 0, 0, -20, -40],
    [-30, 0, 10, 15, 15, 10, 0, -30],
    [-30, 5, 15, 20, 20, 15, 5, -30],
    [-30, 0, 15, 20, 20, 15, 0, -30],
    [-30, 5, 10, 15, 15, 10, 5, -30],
    [-40, -20, 0, 5, 5, 0, -20, -40],
    [-50, -40, -30, -30, -30, -30, -40, -50],
]


BISHOP_TABLE = [
    [-20, -10, -10, -10, -10, -10, -10, -20],
    [-10, 0, 0, 0, 0, 0, 0, -10],
    [-10, 0, 5, 10, 10, 5, 0, -10],
    [-10, 5, 5, 10, 10, 5, 5, -10],
    [-10, 0, 10, 10, 10, 10, 0, -10],
    [-10, 10, 10, 10, 10, 10, 10, -10],
    [-10, 5, 0, 0, 0, 0, 5, -10],
    [-20, -10, -10, -10, -10, -10, -10, -20],
]


ROOK_TABLE = [
    [0, 0, 0, 5, 5, 0, 0, 0],
    [5, 10, 10, 10, 10, 10, 10, 5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [0, 0, 0, 5, 5, 0, 0, 0],
]


QUEEN_TABLE = [
    [-20, -10, -10, -5, -5, -10, -10, -20],
    [-10, 0, 0, 0, 0, 0, 0, -10],
    [-10, 0, 5, 5, 5, 5, 0, -10],
    [-5, 0, 5, 5, 5, 5, 0, -5],
    [0, 0, 5, 5, 5, 5, 0, -5],
    [-10, 5, 5, 5, 5, 5, 0, -10],
    [-10, 0, 5, 0, 0, 0, 0, -10],
    [-20, -10, -10, -5, -5, -10, -10, -20],
]


KING_MIDDLEGAME_TABLE = [
    [20, 30, 10, 0, 0, 10, 30, 20],
    [20, 20, 0, 0, 0, 0, 20, 20],
    [-10, -20, -20, -20, -20, -20, -20, -10],
    [-20, -30, -30, -40, -40, -30, -30, -20],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
]


KING_ENDGAME_TABLE = [
    [-50, -30, -30, -30, -30, -30, -30, -50],
    [-30, -20, -10, -10, -10, -10, -20, -30],
    [-30, -10, 20, 30, 30, 20, -10, -30],
    [-30, -10, 30, 40, 40, 30, -10, -30],
    [-30, -10, 30, 40, 40, 30, -10, -30],
    [-30, -10, 20, 30, 30, 20, -10, -30],
    [-30, -30, 0, 0, 0, 0, -30, -30],
    [-50, -30, -30, -30, -30, -30, -30, -50],
]


POSITION_TABLES = {
    chess.PAWN: PAWN_TABLE,
    chess.KNIGHT: KNIGHT_TABLE,
    chess.BISHOP: BISHOP_TABLE,
    chess.ROOK: ROOK_TABLE,
    chess.QUEEN: QUEEN_TABLE,
}


PASSED_PAWN_BONUS = [0, 5, 10, 20, 35, 60, 100, 0]
CENTER_SQUARES = [chess.D4, chess.E4, chess.D5, chess.E5]

# Máscaras bitboard pré-calculadas. Evitam loops Python repetidos durante
# a avaliação de peões, torres e segurança do rei.
FILE_MASKS = tuple(
    sum(chess.BB_SQUARES[chess.square(file, rank)] for rank in range(8))
    for file in range(8)
)


def _build_passed_pawn_masks(color):
    masks = [0] * 64

    for square in chess.SQUARES:
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        mask = 0

        files = range(max(0, file - 1), min(7, file + 1) + 1)

        if color == chess.WHITE:
            ranks = range(rank + 1, 8)
        else:
            ranks = range(rank - 1, -1, -1)

        for f in files:
            for r in ranks:
                mask |= chess.BB_SQUARES[chess.square(f, r)]

        masks[square] = mask

    return tuple(masks)


PASSED_PAWN_MASKS = {
    chess.WHITE: _build_passed_pawn_masks(chess.WHITE),
    chess.BLACK: _build_passed_pawn_masks(chess.BLACK),
}


# ---------------------------------------------------------------------------
# ZOBRIST INCREMENTAL (Polyglot-compatible)
# ---------------------------------------------------------------------------

_ZOBRIST = chess.polyglot.POLYGLOT_RANDOM_ARRAY
_ZOBRIST_TURN = _ZOBRIST[780]


def _zobrist_piece_key(piece, square):
    # python-chess / Polyglot usa pivot 0=pretas, 1=brancas.
    pivot = 1 if piece.color == chess.WHITE else 0
    piece_index = (piece.piece_type - 1) * 2 + pivot
    return _ZOBRIST[64 * piece_index + square]


def _zobrist_castling_component(board):
    key = 0

    if board.has_kingside_castling_rights(chess.WHITE):
        key ^= _ZOBRIST[768]
    if board.has_queenside_castling_rights(chess.WHITE):
        key ^= _ZOBRIST[769]
    if board.has_kingside_castling_rights(chess.BLACK):
        key ^= _ZOBRIST[770]
    if board.has_queenside_castling_rights(chess.BLACK):
        key ^= _ZOBRIST[771]

    return key


def _zobrist_ep_component(board):
    # Polyglot só inclui o ficheiro de en-passant se o lado a jogar
    # tiver um peão que possa capturar pseudo-legalmente nessa casa.
    if board.ep_square is None:
        return 0

    ep_square = board.ep_square
    ep_file = chess.square_file(ep_square)

    if board.turn == chess.WHITE:
        candidates = (
            ep_square - 9,
            ep_square - 7,
        )
    else:
        candidates = (
            ep_square + 7,
            ep_square + 9,
        )

    for square in candidates:
        if not (0 <= square < 64):
            continue

        # Evita wrap entre ficheiros A/H.
        if abs(chess.square_file(square) - ep_file) != 1:
            continue

        piece = board.piece_at(square)

        if (
            piece is not None
            and piece.color == board.turn
            and piece.piece_type == chess.PAWN
        ):
            return _ZOBRIST[772 + ep_file]

    return 0


def push_with_zobrist(board, move, parent_key):
    """Faz push e calcula o hash do filho em O(1), sem varrer o tabuleiro."""
    key = parent_key

    old_castling = _zobrist_castling_component(board)
    old_ep = _zobrist_ep_component(board)

    if move != chess.Move.null():
        moving_piece = board.piece_at(move.from_square)

        if moving_piece is None:
            # Proteção de segurança; em pesquisa normal isto nunca acontece.
            board.push(move)
            return chess.polyglot.zobrist_hash(board)

        key ^= _zobrist_piece_key(moving_piece, move.from_square)

        is_ep = board.is_en_passant(move)
        is_castling = board.is_castling(move)

        if is_ep:
            capture_square = (
                move.to_square - 8
                if moving_piece.color == chess.WHITE
                else move.to_square + 8
            )
            captured_piece = board.piece_at(capture_square)

            if captured_piece is not None:
                key ^= _zobrist_piece_key(captured_piece, capture_square)

        else:
            captured_piece = board.piece_at(move.to_square)

            if captured_piece is not None:
                key ^= _zobrist_piece_key(captured_piece, move.to_square)

        if is_castling:
            # A engine joga standard chess. Mantemos ainda fallback seguro
            # para qualquer posição inesperada.
            if moving_piece.color == chess.WHITE:
                if move.to_square == chess.G1:
                    rook_from, rook_to = chess.H1, chess.F1
                elif move.to_square == chess.C1:
                    rook_from, rook_to = chess.A1, chess.D1
                else:
                    rook_from = rook_to = None
            else:
                if move.to_square == chess.G8:
                    rook_from, rook_to = chess.H8, chess.F8
                elif move.to_square == chess.C8:
                    rook_from, rook_to = chess.A8, chess.D8
                else:
                    rook_from = rook_to = None

            if rook_from is not None:
                rook = board.piece_at(rook_from)

                if rook is not None:
                    key ^= _zobrist_piece_key(rook, rook_from)
                    key ^= _zobrist_piece_key(rook, rook_to)

        placed_piece = moving_piece

        if move.promotion is not None:
            placed_piece = chess.Piece(
                move.promotion,
                moving_piece.color,
            )

        key ^= _zobrist_piece_key(placed_piece, move.to_square)

    board.push(move)

    # Lado a jogar muda em todos os pushes, incluindo null move.
    key ^= _ZOBRIST_TURN

    # Direitos de roque / EP podem mudar com o lance.
    key ^= old_castling
    key ^= _zobrist_castling_component(board)
    key ^= old_ep
    key ^= _zobrist_ep_component(board)

    return key


@dataclass
class TTEntry:
    key: int
    depth: int
    score: int
    flag: int
    best_move: Optional[chess.Move]
    generation: int


@dataclass
class MoveInfo:
    move: chess.Move
    is_capture: bool
    is_quiet: bool
    gives_check: bool
    order_score: int


# TT fixa 2-way: sem apagar 25% da tabela quando enche.
TT_BUCKET_SIZE = 2
TT_GENERATION = 0
TT_ENTRY_COUNT = 0
TRANSPOSITION_TABLE = [None] * MAX_TT_ENTRIES

KILLER_MOVES = {}
HISTORY_HEURISTIC = {}
COUNTERMOVE_HEURISTIC = {}
CONTINUATION_HISTORY = {}
CAPTURE_HISTORY = {}

# Cache pequeno para não repetir a avaliação neural dos filhos da raiz
# em cada iteração do iterative deepening / aspiration window.
ROOT_AI_SCORE_CACHE = {}
MAX_ROOT_AI_CACHE_ENTRIES = 256

# SEE (Static Exchange Evaluation) cache.
SEE_CACHE = {}
MAX_SEE_CACHE_ENTRIES = 50_000

# Cache da avaliação clássica. A avaliação é uma das partes mais caras da
# engine em Python; transposições e re-searches podem assim reutilizá-la.
STATIC_EVAL_CACHE = {}
MAX_STATIC_EVAL_CACHE_ENTRIES = 50_000

# Cache separado da avaliação material rápida usada por pruning.
FAST_EVAL_CACHE = {}

# Contador incremental de repetições para a pesquisa atual.
# É reconstruído uma vez na raiz e atualizado em O(1) por push/pop.
SEARCH_REPETITION_COUNTS = {}

SEARCH_STATS = {
    "nodes": 0,
    "qnodes": 0,
    "tt_hits": 0,
    "cutoffs": 0,
    "eval_cache_hits": 0,
    "rfp_prunes": 0,
    "lmp_prunes": 0,
    "null_cutoffs": 0,
    "repetition_draws": 0,
    "movegen_nodes": 0,
    "recapture_extensions": 0,
    "root_verifications": 0,
    "root_verification_plies": 0,
    "singular_extensions": 0,
    "check_extensions": 0,
    "passed_pawn_extensions": 0,
    "history_maluses": 0,
    "capture_history_updates": 0,
    "fast_eval_hits": 0,
    "tactical_early_stop_blocks": 0,
}


class SearchTimeout(Exception):
    pass


def _reset_tt_storage():
    global TRANSPOSITION_TABLE
    global TT_ENTRY_COUNT

    TRANSPOSITION_TABLE = [None] * max(
        TT_BUCKET_SIZE,
        MAX_TT_ENTRIES,
    )
    TT_ENTRY_COUNT = 0


def _new_tt_generation():
    global TT_GENERATION
    TT_GENERATION = (TT_GENERATION + 1) & 0xFFFF


def clear_engine_state():
    KILLER_MOVES.clear()
    HISTORY_HEURISTIC.clear()
    COUNTERMOVE_HEURISTIC.clear()
    CONTINUATION_HISTORY.clear()
    CAPTURE_HISTORY.clear()
    ROOT_AI_SCORE_CACHE.clear()
    SEE_CACHE.clear()
    STATIC_EVAL_CACHE.clear()
    FAST_EVAL_CACHE.clear()
    SEARCH_REPETITION_COUNTS.clear()
    _reset_tt_storage()
    clear_ai_cache()



def set_hash_size(megabytes):
    global MAX_TT_ENTRIES

    try:
        megabytes = int(megabytes)
    except (ValueError, TypeError):
        return

    megabytes = max(1, min(1024, megabytes))

    # Mantém a mesma escala histórica da Barromax para comparação.
    requested_entries = max(
        TT_BUCKET_SIZE,
        megabytes * 3000,
    )

    # A TT deve ter um número inteiro de buckets.
    MAX_TT_ENTRIES = (
        requested_entries // TT_BUCKET_SIZE
    ) * TT_BUCKET_SIZE

    _reset_tt_storage()


def reset_search_stats():
    SEARCH_STATS["nodes"] = 0
    SEARCH_STATS["qnodes"] = 0
    SEARCH_STATS["tt_hits"] = 0
    SEARCH_STATS["cutoffs"] = 0
    SEARCH_STATS["eval_cache_hits"] = 0
    SEARCH_STATS["rfp_prunes"] = 0
    SEARCH_STATS["lmp_prunes"] = 0
    SEARCH_STATS["null_cutoffs"] = 0
    SEARCH_STATS["repetition_draws"] = 0
    SEARCH_STATS["movegen_nodes"] = 0
    SEARCH_STATS["recapture_extensions"] = 0
    SEARCH_STATS["root_verifications"] = 0
    SEARCH_STATS["root_verification_plies"] = 0
    SEARCH_STATS["singular_extensions"] = 0
    SEARCH_STATS["check_extensions"] = 0
    SEARCH_STATS["passed_pawn_extensions"] = 0
    SEARCH_STATS["history_maluses"] = 0
    SEARCH_STATS["capture_history_updates"] = 0
    SEARCH_STATS["fast_eval_hits"] = 0
    SEARCH_STATS["tactical_early_stop_blocks"] = 0



def get_search_stats():
    stats = dict(SEARCH_STATS)
    stats["tt_entries"] = TT_ENTRY_COUNT
    stats["max_tt_entries"] = MAX_TT_ENTRIES
    return stats



def check_time(deadline):
    if deadline is not None and time.perf_counter() >= deadline:
        raise SearchTimeout()


def get_board_key(board):
    # Usado apenas quando não temos já o hash incremental do nó.
    return chess.polyglot.zobrist_hash(board)


def prepare_search_repetitions(board):
    """Reconstrói uma vez o histórico e depois usa hashes incrementais."""
    SEARCH_REPETITION_COUNTS.clear()

    replay = board.root()
    key = get_board_key(replay)
    SEARCH_REPETITION_COUNTS[key] = 1

    for move in board.move_stack:
        key = push_with_zobrist(
            replay,
            move,
            key,
        )
        SEARCH_REPETITION_COUNTS[key] = (
            SEARCH_REPETITION_COUNTS.get(key, 0) + 1
        )


def repetition_count(node_key):
    return SEARCH_REPETITION_COUNTS.get(node_key, 0)


def push_search_move(
    board,
    move,
    parent_key,
    track_repetition=True,
):
    """Push + Zobrist incremental + contador O(1) de repetição."""
    child_key = push_with_zobrist(
        board,
        move,
        parent_key,
    )

    if track_repetition:
        SEARCH_REPETITION_COUNTS[child_key] = (
            SEARCH_REPETITION_COUNTS.get(child_key, 0) + 1
        )

    return child_key


def pop_search_move(board, child_key, track_repetition=True):
    if track_repetition:
        count = SEARCH_REPETITION_COUNTS.get(child_key, 0)

        if count <= 1:
            SEARCH_REPETITION_COUNTS.pop(child_key, None)
        else:
            SEARCH_REPETITION_COUNTS[child_key] = count - 1

    board.pop()


def terminal_from_legal_moves(
    board,
    legal_moves,
    ply,
    node_key,
    track_repetition=True,
    in_check=None,
):
    """Terminal/draw usando a lista legal já gerada neste nó."""
    if not legal_moves:
        if in_check is None:
            in_check = board.is_check()

        if in_check:
            return -MATE_SCORE + ply

        return 0

    # Material insuficiente é barato e não depende do histórico.
    if board.is_insufficient_material():
        return 0

    if track_repetition and repetition_count(node_key) >= 3:
        SEARCH_STATS["repetition_draws"] += 1
        return 0

    # A Barromax aceita/assume a reclamação de 50 lances assim que
    # a posição atual atinge 100 halfmoves sem captura/peão.
    if board.halfmove_clock >= 100:
        return 0

    return None


def get_game_phase(board):
    phase = 0

    for piece in board.piece_map().values():
        phase += PHASE_VALUES[piece.piece_type]

    return min(24, phase) / 24


def table_value(table, square, color):
    file = chess.square_file(square)
    rank = chess.square_rank(square)

    if color == chess.WHITE:
        return table[rank][file]

    return table[7 - rank][file]


def get_position_value(piece, square, phase):
    if piece.piece_type == chess.KING:
        middlegame = table_value(KING_MIDDLEGAME_TABLE, square, piece.color)
        endgame = table_value(KING_ENDGAME_TABLE, square, piece.color)
        return int(middlegame * phase + endgame * (1 - phase))

    table = POSITION_TABLES[piece.piece_type]
    return table_value(table, square, piece.color)


def is_terminal_draw(board):
    return (
        board.is_stalemate()
        or board.is_insufficient_material()
        or board.is_seventyfive_moves()
        or board.is_fivefold_repetition()
        or board.is_repetition(3)
        or board.can_claim_fifty_moves()
    )


def terminal_score_for_side_to_move(board, ply):
    if board.is_checkmate():
        return -MATE_SCORE + ply

    if is_terminal_draw(board):
        return 0

    return None


def count_legal_moves(board, color):
    original_turn = board.turn
    board.turn = color

    count = board.legal_moves.count()

    board.turn = original_turn

    return count



def has_pawn_on_square(board, square, color, pawns_mask=None):
    if pawns_mask is None:
        pawns_mask = board.pieces_mask(chess.PAWN, color)

    return bool(pawns_mask & chess.BB_SQUARES[square])



def is_passed_pawn(board, square, color, enemy_pawns_mask=None):
    if enemy_pawns_mask is None:
        enemy_pawns_mask = board.pieces_mask(chess.PAWN, not color)

    return not bool(
        enemy_pawns_mask
        & PASSED_PAWN_MASKS[color][square]
    )



def is_protected_by_pawn(board, square, color, own_pawns_mask=None):
    if own_pawns_mask is None:
        own_pawns_mask = board.pieces_mask(chess.PAWN, color)

    return bool(
        board.attackers_mask(color, square)
        & own_pawns_mask
    )



def evaluate_pawn_structure(
    board,
    color,
    own_pawns_mask=None,
    enemy_pawns_mask=None,
    occupied=None,
):
    if own_pawns_mask is None:
        own_pawns_mask = board.pieces_mask(chess.PAWN, color)

    if enemy_pawns_mask is None:
        enemy_pawns_mask = board.pieces_mask(chess.PAWN, not color)

    if occupied is None:
        occupied = board.occupied

    if not own_pawns_mask:
        return 0

    score = 0
    files = [
        (own_pawns_mask & FILE_MASKS[file]).bit_count()
        for file in range(8)
    ]

    for count in files:
        if count > 1:
            score -= 12 * (count - 1)

    for square in chess.scan_reversed(own_pawns_mask):
        file = chess.square_file(square)
        rank = chess.square_rank(square)

        if color == chess.WHITE:
            progress = rank
            front_square = square + 8
        else:
            progress = 7 - rank
            front_square = square - 8

        has_left = file > 0 and files[file - 1] > 0
        has_right = file < 7 and files[file + 1] > 0

        if not has_left and not has_right:
            score -= 10

        if is_passed_pawn(
            board,
            square,
            color,
            enemy_pawns_mask,
        ):
            score += PASSED_PAWN_BONUS[progress]

            if is_protected_by_pawn(
                board,
                square,
                color,
                own_pawns_mask,
            ):
                score += 8

        if progress >= 5:
            score += 8

        if (
            0 <= front_square < 64
            and occupied & chess.BB_SQUARES[front_square]
        ):
            score -= 5

    return score



def evaluate_rooks(
    board,
    color,
    own_pawns_mask=None,
    enemy_pawns_mask=None,
):
    score = 0

    if own_pawns_mask is None:
        own_pawns_mask = board.pieces_mask(chess.PAWN, color)

    if enemy_pawns_mask is None:
        enemy_pawns_mask = board.pieces_mask(chess.PAWN, not color)

    rook_mask = board.pieces_mask(chess.ROOK, color)

    for rook_square in chess.scan_reversed(rook_mask):
        file = chess.square_file(rook_square)
        rank = chess.square_rank(rook_square)
        file_mask = FILE_MASKS[file]

        own_pawn_on_file = bool(own_pawns_mask & file_mask)
        enemy_pawn_on_file = bool(enemy_pawns_mask & file_mask)

        if not own_pawn_on_file and not enemy_pawn_on_file:
            score += 22
        elif not own_pawn_on_file and enemy_pawn_on_file:
            score += 12

        if color == chess.WHITE and rank == 6:
            score += 15

        if color == chess.BLACK and rank == 1:
            score += 15

    return score



def evaluate_king_safety(
    board,
    color,
    phase,
    own_pawns_mask=None,
    enemy_pawns_mask=None,
):
    king_square = board.king(color)

    if king_square is None:
        return 0

    if phase < 0.35:
        return 0

    score = 0
    enemy = not color

    if own_pawns_mask is None:
        own_pawns_mask = board.pieces_mask(chess.PAWN, color)

    if enemy_pawns_mask is None:
        enemy_pawns_mask = board.pieces_mask(chess.PAWN, enemy)

    king_file = chess.square_file(king_square)
    king_rank = chess.square_rank(king_square)

    if color == chess.WHITE:
        if king_square in (chess.G1, chess.C1):
            score += 45

        elif king_square == chess.E1:
            score -= 20

            if not (
                board.has_kingside_castling_rights(chess.WHITE)
                or board.has_queenside_castling_rights(chess.WHITE)
            ):
                score -= 20

        front_direction = 1

    else:
        if king_square in (chess.G8, chess.C8):
            score += 45

        elif king_square == chess.E8:
            score -= 20

            if not (
                board.has_kingside_castling_rights(chess.BLACK)
                or board.has_queenside_castling_rights(chess.BLACK)
            ):
                score -= 20

        front_direction = -1

    # Escudo de peões por bitboards.
    front_rank = king_rank + front_direction
    second_rank = king_rank + 2 * front_direction

    for file in range(king_file - 1, king_file + 2):
        if not 0 <= file <= 7:
            continue

        has_shield = False

        if 0 <= front_rank <= 7:
            square = chess.square(file, front_rank)

            if own_pawns_mask & chess.BB_SQUARES[square]:
                score += 14
                has_shield = True

        if 0 <= second_rank <= 7:
            square = chess.square(file, second_rank)

            if own_pawns_mask & chess.BB_SQUARES[square]:
                score += 5
                has_shield = True

        if not has_shield:
            score -= 12

    # Colunas abertas perto do rei.
    for file in range(king_file - 1, king_file + 2):
        if not 0 <= file <= 7:
            continue

        file_mask = FILE_MASKS[file]
        own_pawn_on_file = bool(own_pawns_mask & file_mask)
        enemy_pawn_on_file = bool(enemy_pawns_mask & file_mask)

        if not own_pawn_on_file:
            score -= 10

        if not own_pawn_on_file and not enemy_pawn_on_file:
            score -= 10

    # Ataques à zona do rei. Em vez de sets de squares, agregamos num
    # único bitboard e só depois iteramos pelos atacantes únicos.
    king_zone_mask = 0

    for file_delta in (-1, 0, 1):
        for rank_delta in (-1, 0, 1):
            file = king_file + file_delta
            rank = king_rank + rank_delta

            if 0 <= file <= 7 and 0 <= rank <= 7:
                king_zone_mask |= chess.BB_SQUARES[chess.square(file, rank)]

    enemy_attackers_mask = 0

    for square in chess.scan_reversed(king_zone_mask):
        enemy_attackers_mask |= board.attackers_mask(enemy, square)

    attack_units = 0

    for attacker_square in chess.scan_reversed(enemy_attackers_mask):
        piece_type = board.piece_type_at(attacker_square)

        if piece_type == chess.PAWN:
            attack_units += 1
        elif piece_type == chess.KNIGHT:
            attack_units += 3
        elif piece_type == chess.BISHOP:
            attack_units += 3
        elif piece_type == chess.ROOK:
            attack_units += 5
        elif piece_type == chess.QUEEN:
            attack_units += 8

    score -= attack_units * 4

    attacker_count = enemy_attackers_mask.bit_count()

    if attacker_count >= 3:
        score -= 15

    if attacker_count >= 4:
        score -= 20

    return int(score * phase)


def evaluate_knight_stability(board, color, phase):
    # Só interessa realmente na abertura / middlegame.
    if phase < 0.55:
        return 0

    score = 0
    enemy_color = not color

    for square in board.pieces(chess.KNIGHT, color):

        # Verifica se algum peão inimigo ataca o cavalo.
        attacked_by_enemy_pawn = False

        for attacker_square in board.attackers(enemy_color, square):
            attacker = board.piece_at(attacker_square)

            if (
                attacker is not None
                and attacker.piece_type == chess.PAWN
                and attacker.color == enemy_color
            ):
                attacked_by_enemy_pawn = True
                break

        if attacked_by_enemy_pawn:
            # Cavalo pode ser expulso com ganho de tempo.
            score -= 30

            # Penalização adicional se estiver muito avançado/centralizado.
            if square in [
                chess.C4, chess.D4, chess.E4, chess.F4,
                chess.C5, chess.D5, chess.E5, chess.F5,
            ]:
                score -= 15

    return score





def pawn_can_chase_knight(
    board,
    knight_square,
    knight_color,
    enemy_pawns_mask=None,
    occupied=None,
):
    enemy_color = not knight_color

    if enemy_pawns_mask is None:
        enemy_pawns_mask = board.pieces_mask(chess.PAWN, enemy_color)

    if occupied is None:
        occupied = board.occupied

    knight_file = chess.square_file(knight_square)
    knight_rank = chess.square_rank(knight_square)

    for pawn_square in chess.scan_reversed(enemy_pawns_mask):
        pawn_file = chess.square_file(pawn_square)
        pawn_rank = chess.square_rank(pawn_square)

        if enemy_color == chess.WHITE:
            direction = 1
            start_rank = 1
        else:
            direction = -1
            start_rank = 6

        advances = (1, 2) if pawn_rank == start_rank else (1,)

        for steps in advances:
            new_rank = pawn_rank + direction * steps

            if not 0 <= new_rank <= 7:
                continue

            blocked = False

            for step in range(1, steps + 1):
                intermediate_square = chess.square(
                    pawn_file,
                    pawn_rank + direction * step,
                )

                if occupied & chess.BB_SQUARES[intermediate_square]:
                    blocked = True
                    break

            if blocked:
                continue

            attack_rank = new_rank + direction

            if not 0 <= attack_rank <= 7:
                continue

            if (
                attack_rank == knight_rank
                and abs(pawn_file - knight_file) == 1
            ):
                return True

    return False



def evaluate_knight_pressure(
    board,
    color,
    phase,
    enemy_pawns_mask=None,
    occupied=None,
):
    if phase < 0.55:
        return 0

    score = 0
    enemy_color = not color

    if enemy_pawns_mask is None:
        enemy_pawns_mask = board.pieces_mask(chess.PAWN, enemy_color)

    if occupied is None:
        occupied = board.occupied

    knight_mask = board.pieces_mask(chess.KNIGHT, color)

    for square in chess.scan_reversed(knight_mask):
        attacked_by_pawn = bool(
            board.attackers_mask(enemy_color, square)
            & enemy_pawns_mask
        )

        if attacked_by_pawn:
            score -= 35

        elif pawn_can_chase_knight(
            board,
            square,
            color,
            enemy_pawns_mask,
            occupied,
        ):
            score -= 20

        rank = chess.square_rank(square)

        if color == chess.WHITE:
            if rank >= 4:
                score -= 8
        else:
            if rank <= 3:
                score -= 8

    return score


def evaluate_development(board, color, phase):
    if phase < 0.65:
        return 0

    score = 0

    if color == chess.WHITE:
        starting_minors = [chess.B1, chess.G1, chess.C1, chess.F1]
        queen_start = chess.D1
    else:
        starting_minors = [chess.B8, chess.G8, chess.C8, chess.F8]
        queen_start = chess.D8

    for square in starting_minors:
        piece = board.piece_at(square)

        if piece is not None and piece.color == color and piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
            score -= 12

    queens = list(board.pieces(chess.QUEEN, color))

    if queens:
        queen_square = queens[0]

        if queen_square != queen_start and board.fullmove_number <= 10:
            score -= 10

    return score



def evaluate_center_control(board):
    score = 0

    for square in CENTER_SQUARES:
        white_attackers = board.attackers_mask(chess.WHITE, square).bit_count()
        black_attackers = board.attackers_mask(chess.BLACK, square).bit_count()

        score += 3 * (white_attackers - black_attackers)

    return score



def evaluate_classic_board(board, assume_nonterminal=False):
    """
    Positivo = vantagem das brancas.
    Negativo = vantagem das pretas.

    Durante a pesquisa, assume_nonterminal=True evita repetir testes caros
    de mate/repetição que o search já fez antes de chamar a avaliação.
    """

    if not assume_nonterminal:
        if board.is_checkmate():
            if board.turn == chess.WHITE:
                return -MATE_SCORE
            return MATE_SCORE

        if is_terminal_draw(board):
            return 0

    score = 0

    piece_map = board.piece_map()
    phase_units = sum(
        PHASE_VALUES[piece.piece_type]
        for piece in piece_map.values()
    )
    phase = min(24, phase_units) / 24

    white_pawns = board.pieces_mask(chess.PAWN, chess.WHITE)
    black_pawns = board.pieces_mask(chess.PAWN, chess.BLACK)
    occupied = board.occupied

    for square, piece in piece_map.items():
        material = PIECE_VALUES[piece.piece_type]
        position = get_position_value(piece, square, phase)
        total = material + position

        if piece.color == chess.WHITE:
            score += total
        else:
            score -= total

    if board.pieces_mask(chess.BISHOP, chess.WHITE).bit_count() >= 2:
        score += 35

    if board.pieces_mask(chess.BISHOP, chess.BLACK).bit_count() >= 2:
        score -= 35

    score += evaluate_pawn_structure(
        board,
        chess.WHITE,
        white_pawns,
        black_pawns,
        occupied,
    )
    score -= evaluate_pawn_structure(
        board,
        chess.BLACK,
        black_pawns,
        white_pawns,
        occupied,
    )

    score += evaluate_rooks(
        board,
        chess.WHITE,
        white_pawns,
        black_pawns,
    )
    score -= evaluate_rooks(
        board,
        chess.BLACK,
        black_pawns,
        white_pawns,
    )

    score += evaluate_king_safety(
        board,
        chess.WHITE,
        phase,
        white_pawns,
        black_pawns,
    )
    score -= evaluate_king_safety(
        board,
        chess.BLACK,
        phase,
        black_pawns,
        white_pawns,
    )

    score += evaluate_development(board, chess.WHITE, phase)
    score -= evaluate_development(board, chess.BLACK, phase)

    score += evaluate_knight_pressure(
        board,
        chess.WHITE,
        phase,
        black_pawns,
        occupied,
    )
    score -= evaluate_knight_pressure(
        board,
        chess.BLACK,
        phase,
        white_pawns,
        occupied,
    )

    score += evaluate_center_control(board)

    if board.turn == chess.WHITE:
        score += 8
    else:
        score -= 8

    return int(score)



def evaluate_board(board, assume_nonterminal=False):
    """
    Avaliação final da Barromax.

    A rede neural continua usada para ordering na raiz. A avaliação interna
    da árvore permanece clássica para ser previsível e rápida.
    """
    return evaluate_classic_board(
        board,
        assume_nonterminal=assume_nonterminal,
    )



def evaluate_for_side_to_move(board, assume_nonterminal=False):
    score = evaluate_board(
        board,
        assume_nonterminal=assume_nonterminal,
    )

    if board.turn == chess.WHITE:
        return score

    return -score



def static_eval_cache_key(board, node_key=None):
    if node_key is None:
        node_key = get_board_key(board)

    return (
        node_key,
        board.halfmove_clock,
        board.fullmove_number,
    )



def get_static_eval(board, node_key=None):
    """Avaliação completa do ponto de vista de quem joga, com cache limitado."""
    key = static_eval_cache_key(board, node_key)
    cached = STATIC_EVAL_CACHE.get(key)

    if cached is not None:
        SEARCH_STATS["eval_cache_hits"] += 1
        return cached

    # Todos os call-sites do search verificam terminal antes de chegar aqui.
    score = evaluate_for_side_to_move(
        board,
        assume_nonterminal=True,
    )

    if len(STATIC_EVAL_CACHE) >= MAX_STATIC_EVAL_CACHE_ENTRIES:
        STATIC_EVAL_CACHE.clear()

    STATIC_EVAL_CACHE[key] = score
    return score


def evaluate_fast_for_side_to_move(board):
    """
    Avaliação muito barata para decisões de pruning.

    Usa apenas material, par de bispos e tempo. Não substitui a avaliação
    completa nas folhas: serve para evitar chamar a avaliação cara em nós
    que estão claramente longe da margem de corte.
    """
    white_score = 0

    for piece_type in (
        chess.PAWN,
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
        chess.QUEEN,
    ):
        value = PIECE_VALUES[piece_type]
        white_count = board.pieces_mask(piece_type, chess.WHITE).bit_count()
        black_count = board.pieces_mask(piece_type, chess.BLACK).bit_count()
        white_score += value * (white_count - black_count)

    if board.pieces_mask(chess.BISHOP, chess.WHITE).bit_count() >= 2:
        white_score += 30

    if board.pieces_mask(chess.BISHOP, chess.BLACK).bit_count() >= 2:
        white_score -= 30

    white_score += 8 if board.turn == chess.WHITE else -8

    if board.turn == chess.WHITE:
        return int(white_score)

    return int(-white_score)


def get_fast_eval(board, node_key=None):
    if node_key is None:
        node_key = get_board_key(board)

    key = (
        node_key,
        board.halfmove_clock,
        board.fullmove_number,
    )

    cached = FAST_EVAL_CACHE.get(key)

    if cached is not None:
        SEARCH_STATS["fast_eval_hits"] += 1
        return cached

    score = evaluate_fast_for_side_to_move(board)

    if len(FAST_EVAL_CACHE) >= MAX_FAST_EVAL_CACHE_ENTRIES:
        FAST_EVAL_CACHE.clear()

    FAST_EVAL_CACHE[key] = score
    return score



def history_key(move):
    return (move.from_square, move.to_square, move.promotion)


def _bounded_history_update(table, key, bonus):
    bonus = max(-HISTORY_MAX, min(HISTORY_MAX, int(bonus)))
    old = table.get(key, 0)
    new = old + bonus - (old * abs(bonus) // HISTORY_MAX)
    table[key] = max(-HISTORY_MAX, min(HISTORY_MAX, int(new)))
    return table[key]


def continuation_history_key(previous_move, move):
    if previous_move is None:
        return None

    return (
        history_key(previous_move),
        history_key(move),
    )


def quiet_history_score(move, previous_move=None):
    score = HISTORY_HEURISTIC.get(history_key(move), 0)

    continuation_key = continuation_history_key(previous_move, move)
    if continuation_key is not None:
        score += CONTINUATION_HISTORY.get(continuation_key, 0)

        counter = COUNTERMOVE_HEURISTIC.get(history_key(previous_move))
        if counter == move:
            score += COUNTERMOVE_BONUS

    return score


def update_history(move, depth, previous_move=None):
    bonus = min(4_000, 32 * depth * depth)
    _bounded_history_update(HISTORY_HEURISTIC, history_key(move), bonus)

    continuation_key = continuation_history_key(previous_move, move)
    if continuation_key is not None:
        _bounded_history_update(CONTINUATION_HISTORY, continuation_key, bonus)
        COUNTERMOVE_HEURISTIC[history_key(previous_move)] = move


def penalize_history(move, depth, previous_move=None):
    malus = -min(2_000, 16 * depth * depth)
    _bounded_history_update(HISTORY_HEURISTIC, history_key(move), malus)

    continuation_key = continuation_history_key(previous_move, move)
    if continuation_key is not None:
        _bounded_history_update(CONTINUATION_HISTORY, continuation_key, malus)

    SEARCH_STATS["history_maluses"] += 1


def capture_history_key(board, move):
    attacker = board.piece_at(move.from_square)

    if attacker is None:
        return None

    if board.is_en_passant(move):
        victim_type = chess.PAWN
    else:
        victim = board.piece_at(move.to_square)
        victim_type = victim.piece_type if victim is not None else 0

    return (
        attacker.piece_type,
        move.to_square,
        victim_type,
        move.promotion,
    )


def update_capture_history(board, move, depth, success=True):
    key = capture_history_key(board, move)

    if key is None:
        return

    magnitude = min(3_000, 24 * max(1, depth) * max(1, depth))
    bonus = magnitude if success else -magnitude // 2
    _bounded_history_update(CAPTURE_HISTORY, key, bonus)
    SEARCH_STATS["capture_history_updates"] += 1



def add_killer_move(ply, move):
    killers = KILLER_MOVES.get(ply, [])

    if move in killers:
        return

    killers.insert(0, move)
    KILLER_MOVES[ply] = killers[:2]


def see_piece_value(piece_type):
    # O rei recebe um valor enorme apenas para o SEE; nunca é uma
    # peça que queremos "trocar" materialmente.
    if piece_type == chess.KING:
        return 20_000

    return PIECE_VALUES.get(piece_type, 0)



def least_valuable_legal_attacker(board, target_square):
    """
    Atacante legal de menor valor usando o gerador de capturas do
    python-chess por máscara de origem. Evita construir/testar manualmente
    cada candidato com board.is_legal().
    """
    target_mask = chess.BB_SQUARES[target_square]

    for piece_type in (
        chess.PAWN,
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
        chess.QUEEN,
        chess.KING,
    ):
        from_mask = board.pieces_mask(piece_type, board.turn)

        if not from_mask:
            continue

        fallback = None

        for candidate in board.generate_legal_captures(
            from_mask=from_mask,
            to_mask=target_mask,
        ):
            # Em promoção, preferimos dama para o cálculo material.
            if candidate.promotion == chess.QUEEN:
                return candidate

            if fallback is None:
                fallback = candidate

        if fallback is not None:
            return fallback

    return None



def static_exchange_eval(board, move, board_key=None):
    """
    Static Exchange Evaluation (SEE).

    Versão 2.3:
    - reutiliza o hash do nó quando disponível;
    - evita board.copy();
    - usa generate_legal_captures() para encontrar recapturas legais;
    - restaura sempre o board original com push/pop.
    """
    if not board.is_capture(move):
        return 0

    if board_key is None:
        board_key = get_board_key(board)

    cache_key = (
        board_key,
        move.from_square,
        move.to_square,
        move.promotion,
    )

    cached = SEE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if board.is_en_passant(move):
        captured_value = PIECE_VALUES[chess.PAWN]
    else:
        captured_piece = board.piece_at(move.to_square)
        captured_value = (
            PIECE_VALUES[captured_piece.piece_type]
            if captured_piece is not None
            else 0
        )

    gains = [captured_value]

    if move.promotion is not None:
        gains[0] += (
            PIECE_VALUES.get(move.promotion, 0)
            - PIECE_VALUES[chess.PAWN]
        )

    target_square = move.to_square
    pushed = 0

    try:
        board.push(move)
        pushed += 1

        while True:
            piece_on_target = board.piece_at(target_square)

            if piece_on_target is None:
                break

            recapture = least_valuable_legal_attacker(
                board,
                target_square,
            )

            if recapture is None:
                break

            exchange_gain = (
                see_piece_value(piece_on_target.piece_type)
                - gains[-1]
            )

            if recapture.promotion is not None:
                exchange_gain += (
                    PIECE_VALUES.get(recapture.promotion, 0)
                    - PIECE_VALUES[chess.PAWN]
                )

            gains.append(exchange_gain)
            board.push(recapture)
            pushed += 1

    finally:
        for _ in range(pushed):
            board.pop()

    for index in range(len(gains) - 1, 0, -1):
        gains[index - 1] = -max(
            -gains[index - 1],
            gains[index],
        )

    result = int(gains[0])

    if len(SEE_CACHE) >= MAX_SEE_CACHE_ENTRIES:
        SEE_CACHE.clear()

    SEE_CACHE[cache_key] = result
    return result



def capture_score(
    board,
    move,
    board_key=None,
    gives_check=None,
):
    attacker = board.piece_at(move.from_square)

    if board.is_en_passant(move):
        victim_value = PIECE_VALUES[chess.PAWN]
    else:
        victim = board.piece_at(move.to_square)
        victim_value = PIECE_VALUES[victim.piece_type] if victim else 0

    attacker_value = PIECE_VALUES[attacker.piece_type] if attacker else 0

    defended_target = board.is_attacked_by(
        not board.turn,
        move.to_square,
    )

    suspicious_capture = (
        attacker_value > victim_value
        or defended_target
        or board.is_en_passant(move)
    )

    if suspicious_capture:
        see_score = static_exchange_eval(
            board,
            move,
            board_key=board_key,
        )
    else:
        see_score = victim_value

    # gives_check é calculado uma única vez no move picker e reutilizado.
    if gives_check is None:
        gives_check = board.gives_check(move)

    base = 100_000

    if see_score < 0 and not gives_check:
        base = 20_000

    history_bonus = 0
    capture_key = capture_history_key(board, move)
    if capture_key is not None:
        history_bonus = CAPTURE_HISTORY.get(capture_key, 0)

    return (
        base
        + 10 * victim_value
        - attacker_value
        + 50 * max(-1000, min(1000, see_score))
        + history_bonus
    )




def is_quiet_move(board, move, is_capture=None):
    if is_capture is None:
        is_capture = board.is_capture(move)

    return not is_capture and move.promotion is None



def move_order_score(
    board,
    move,
    ply=0,
    tt_move=None,
    board_key=None,
    is_capture=None,
    gives_check=None,
    capture_component=None,
):
    if tt_move is not None and move == tt_move:
        return 1_000_000

    if is_capture is None:
        is_capture = board.is_capture(move)

    if gives_check is None:
        gives_check = board.gives_check(move)

    score = 0

    if move.promotion:
        score += 900_000 + PIECE_VALUES.get(move.promotion, 0)

    if is_capture:
        if capture_component is None:
            capture_component = capture_score(
                board,
                move,
                board_key=board_key,
                gives_check=gives_check,
            )

        score += capture_component

    elif move in KILLER_MOVES.get(ply, []):
        score += 80_000

    else:
        score += HISTORY_HEURISTIC.get(history_key(move), 0)

    if gives_check:
        score += 5_000

    if board.is_castling(move):
        score += 2_000

    piece = board.piece_at(move.from_square)

    if piece is not None:
        if piece.piece_type in (chess.KNIGHT, chess.BISHOP):
            from_rank = chess.square_rank(move.from_square)

            if piece.color == chess.WHITE and from_rank == 0:
                score += 200
            elif piece.color == chess.BLACK and from_rank == 7:
                score += 200

        if move.to_square in CENTER_SQUARES:
            if piece.piece_type == chess.PAWN:
                score += 140
            elif piece.piece_type == chess.KNIGHT:
                score += 15
            else:
                score += 40

    return score




def _quiet_order_score_no_check(
    board,
    move,
    ply,
    previous_move=None,
):
    """Ordering barato para quiet moves sem calcular gives_check."""
    score = quiet_history_score(move, previous_move)

    if move in KILLER_MOVES.get(ply, []):
        score += 80_000

    if board.is_castling(move):
        score += 2_000

    piece = board.piece_at(move.from_square)

    if piece is not None:
        if piece.piece_type in (
            chess.KNIGHT,
            chess.BISHOP,
        ):
            from_rank = chess.square_rank(
                move.from_square
            )

            if (
                piece.color == chess.WHITE
                and from_rank == 0
            ):
                score += 200
            elif (
                piece.color == chess.BLACK
                and from_rank == 7
            ):
                score += 200

        if move.to_square in CENTER_SQUARES:
            if piece.piece_type == chess.PAWN:
                score += 140
            elif piece.piece_type == chess.KNIGHT:
                score += 15
            else:
                score += 40

    return score



def iter_ordered_move_infos(
    board,
    legal_moves,
    ply=0,
    tt_move=None,
    board_key=None,
):
    """
    Move picker 2.4 deep-search.

    Fases:
    TT -> promoções -> boas capturas -> killers/countermove
    -> quiet moves por history/continuation -> capturas SEE-negativas.
    """
    if board_key is None:
        board_key = get_board_key(board)

    legal_moves = list(legal_moves)
    legal_set = set(legal_moves)
    previous_move = board.peek() if board.move_stack else None

    if tt_move is not None and tt_move not in legal_set:
        tt_move = None

    if tt_move is not None:
        is_capture = board.is_capture(tt_move)
        gives_check = board.gives_check(tt_move)

        yield MoveInfo(
            move=tt_move,
            is_capture=is_capture,
            is_quiet=(not is_capture and tt_move.promotion is None),
            gives_check=gives_check,
            order_score=1_000_000,
        )

    promotions = []
    good_captures = []
    bad_captures = []
    killer_moves = []
    quiet_moves = []

    killer_set = set(KILLER_MOVES.get(ply, []))
    counter_move = None

    if previous_move is not None:
        counter_move = COUNTERMOVE_HEURISTIC.get(history_key(previous_move))

    for move in legal_moves:
        if tt_move is not None and move == tt_move:
            continue

        is_capture = board.is_capture(move)

        if move.promotion is not None:
            gives_check = board.gives_check(move)
            score = 900_000 + PIECE_VALUES.get(move.promotion, 0)

            if is_capture:
                victim = board.piece_at(move.to_square)
                if victim is not None:
                    score += 10 * PIECE_VALUES[victim.piece_type]

            if gives_check:
                score += 5_000

            promotions.append(
                MoveInfo(move, is_capture, False, gives_check, score)
            )
            continue

        if is_capture:
            gives_check = board.gives_check(move)
            capture_component = capture_score(
                board,
                move,
                board_key=board_key,
                gives_check=gives_check,
            )
            score = capture_component + (5_000 if gives_check else 0)
            info = MoveInfo(move, True, False, gives_check, score)

            if capture_component >= 100_000:
                good_captures.append(info)
            else:
                bad_captures.append(info)
            continue

        cheap_score = _quiet_order_score_no_check(
            board,
            move,
            ply,
            previous_move=previous_move,
        )

        if move in killer_set or move == counter_move:
            killer_moves.append((cheap_score, move))
        else:
            quiet_moves.append((cheap_score, move))

    promotions.sort(key=lambda item: item.order_score, reverse=True)
    good_captures.sort(key=lambda item: item.order_score, reverse=True)
    bad_captures.sort(key=lambda item: item.order_score, reverse=True)
    killer_moves.sort(key=lambda item: item[0], reverse=True)
    quiet_moves.sort(key=lambda item: item[0], reverse=True)

    for info in promotions:
        yield info

    for info in good_captures:
        yield info

    for cheap_score, move in killer_moves:
        gives_check = board.gives_check(move)
        yield MoveInfo(
            move=move,
            is_capture=False,
            is_quiet=True,
            gives_check=gives_check,
            order_score=cheap_score + (5_000 if gives_check else 0),
        )

    for cheap_score, move in quiet_moves:
        gives_check = board.gives_check(move)
        yield MoveInfo(
            move=move,
            is_capture=False,
            is_quiet=True,
            gives_check=gives_check,
            order_score=cheap_score + (5_000 if gives_check else 0),
        )

    for info in bad_captures:
        yield info



def order_moves_from_list(
    board,
    legal_moves,
    ply=0,
    tt_move=None,
    board_key=None,
):
    # Compatibilidade com a raiz/AI, onde precisamos da lista completa.
    return list(
        iter_ordered_move_infos(
            board,
            legal_moves,
            ply=ply,
            tt_move=tt_move,
            board_key=board_key,
        )
    )


def order_moves(
    board,
    ply=0,
    tt_move=None,
    board_key=None,
):
    legal_moves = list(board.legal_moves)

    return [
        info.move
        for info in iter_ordered_move_infos(
            board,
            legal_moves,
            ply=ply,
            tt_move=tt_move,
            board_key=board_key,
        )
    ]


def _direct_qsearch_candidates(
    board,
    include_quiet_checks,
):
    """
    Gera diretamente capturas e promoções.

    Só no primeiro nível da qsearch acrescentamos quiet checks; nos níveis
    internos evitamos percorrer todos os lances legais.
    """
    moves = []
    seen = set()

    for move in board.generate_legal_captures():
        if move not in seen:
            seen.add(move)
            moves.append(move)

    promotion_rank = (
        chess.BB_RANK_8
        if board.turn == chess.WHITE
        else chess.BB_RANK_1
    )

    for move in board.generate_legal_moves(
        from_mask=board.pieces_mask(
            chess.PAWN,
            board.turn,
        ),
        to_mask=promotion_rank,
    ):
        if move.promotion is None:
            continue

        if move not in seen:
            seen.add(move)
            moves.append(move)

    if include_quiet_checks:
        for move in board.generate_legal_moves():
            if move in seen:
                continue

            if board.gives_check(move):
                seen.add(move)
                moves.append(move)

    return moves


def order_tactical_move_infos(
    board,
    legal_moves,
    q_depth,
    board_key=None,
):
    """
    Ordena apenas lances táticos.

    Se legal_moves=None, usa geração direta de capturas/promoções e evita
    gerar toda a lista legal nos níveis internos da qsearch.
    """
    if board_key is None:
        board_key = get_board_key(board)

    include_checks = (
        q_depth == QUIESCENCE_DEPTH
    )

    if legal_moves is None:
        legal_moves = _direct_qsearch_candidates(
            board,
            include_quiet_checks=include_checks,
        )

    infos = []

    for move in legal_moves:
        is_capture = board.is_capture(move)
        promotion = move.promotion is not None

        if (
            not is_capture
            and not promotion
            and not include_checks
        ):
            continue

        gives_check = board.gives_check(move)

        if (
            not is_capture
            and not promotion
            and not gives_check
        ):
            continue

        capture_component = None

        if is_capture:
            capture_component = capture_score(
                board,
                move,
                board_key=board_key,
                gives_check=gives_check,
            )

        score = move_order_score(
            board,
            move,
            ply=0,
            tt_move=None,
            board_key=board_key,
            is_capture=is_capture,
            gives_check=gives_check,
            capture_component=capture_component,
        )

        infos.append(
            MoveInfo(
                move=move,
                is_capture=is_capture,
                is_quiet=(
                    not is_capture
                    and move.promotion is None
                ),
                gives_check=gives_check,
                order_score=score,
            )
        )

    infos.sort(
        key=lambda item: item.order_score,
        reverse=True,
    )

    return infos



def _tt_bucket_base(key):
    bucket_count = max(
        1,
        len(TRANSPOSITION_TABLE) // TT_BUCKET_SIZE,
    )
    return (key % bucket_count) * TT_BUCKET_SIZE


def get_tt_entry(key):
    base = _tt_bucket_base(key)

    for index in range(base, base + TT_BUCKET_SIZE):
        entry = TRANSPOSITION_TABLE[index]

        if entry is not None and entry.key == key:
            return entry

    return None


def probe_tt(board, depth, alpha, beta, key=None):
    if key is None:
        key = get_board_key(board)

    entry = get_tt_entry(key)

    if entry is None:
        return None, None

    tt_move = entry.best_move

    if entry.depth >= depth:
        if entry.flag == TT_EXACT:
            SEARCH_STATS["tt_hits"] += 1
            return entry.score, tt_move

        if entry.flag == TT_LOWER and entry.score >= beta:
            SEARCH_STATS["tt_hits"] += 1
            return entry.score, tt_move

        if entry.flag == TT_UPPER and entry.score <= alpha:
            SEARCH_STATS["tt_hits"] += 1
            return entry.score, tt_move

    return None, tt_move



def store_tt(
    board,
    depth,
    score,
    flag,
    best_move,
    key=None,
):
    global TT_ENTRY_COUNT

    if key is None:
        key = get_board_key(board)

    base = _tt_bucket_base(key)
    matching_index = None
    empty_index = None
    replacement_index = None
    replacement_rank = None

    for index in range(base, base + TT_BUCKET_SIZE):
        entry = TRANSPOSITION_TABLE[index]

        if entry is None:
            if empty_index is None:
                empty_index = index
            continue

        if entry.key == key:
            matching_index = index
            break

        # Preferir substituir entradas de gerações anteriores; dentro da
        # mesma geração, substituir a de menor profundidade.
        age_penalty = (
            1000
            if entry.generation != TT_GENERATION
            else 0
        )
        rank = entry.depth - age_penalty

        if (
            replacement_rank is None
            or rank < replacement_rank
        ):
            replacement_rank = rank
            replacement_index = index

    target_index = matching_index

    if target_index is not None:
        old_entry = TRANSPOSITION_TABLE[target_index]

        # Preserva uma entrada significativamente mais profunda, exceto se
        # a nova for EXACT.
        if (
            old_entry.depth > depth
            and flag != TT_EXACT
        ):
            return

    elif empty_index is not None:
        target_index = empty_index
        TT_ENTRY_COUNT += 1

    else:
        target_index = replacement_index

    TRANSPOSITION_TABLE[target_index] = TTEntry(
        key=key,
        depth=depth,
        score=int(score),
        flag=flag,
        best_move=best_move,
        generation=TT_GENERATION,
    )



def quiescence(
    board,
    alpha,
    beta,
    deadline=None,
    q_depth=QUIESCENCE_DEPTH,
    ply=0,
    node_key=None,
    track_repetition=True,
):
    check_time(deadline)
    SEARCH_STATS["qnodes"] += 1

    if node_key is None:
        node_key = get_board_key(board)

    # Draws baratos antes de qualquer geração extensa de lances.
    if board.is_insufficient_material():
        return 0

    if (
        track_repetition
        and repetition_count(node_key) >= 3
    ):
        SEARCH_STATS["repetition_draws"] += 1
        return 0

    if board.halfmove_clock >= 100:
        return 0

    in_check = board.is_check()

    # Em xeque temos de ver TODAS as evasões, mesmo se q_depth chegou a 0.
    # Isto reduz o efeito horizonte em sequências forçadas de checks.
    if in_check:
        legal_moves = list(
            board.generate_legal_moves()
        )
        SEARCH_STATS["movegen_nodes"] += 1

        if not legal_moves:
            return -MATE_SCORE + ply

        best_score = -INFINITY

        move_infos = iter_ordered_move_infos(
            board,
            legal_moves,
            ply=ply,
            tt_move=None,
            board_key=node_key,
        )

        for info in move_infos:
            check_time(deadline)

            child_key = push_search_move(
                board,
                info.move,
                node_key,
                track_repetition=track_repetition,
            )

            try:
                score = -quiescence(
                    board,
                    -beta,
                    -alpha,
                    deadline,
                    q_depth - 1,
                    ply + 1,
                    node_key=child_key,
                    track_repetition=track_repetition,
                )
            finally:
                pop_search_move(
                    board,
                    child_key,
                    track_repetition=track_repetition,
                )

            if score > best_score:
                best_score = score

            if best_score > alpha:
                alpha = best_score

            if alpha >= beta:
                SEARCH_STATS["cutoffs"] += 1
                return alpha

        return best_score

    # Fora de xeque basta confirmar que existe pelo menos um lance legal
    # para distinguir uma posição normal de stalemate. O generator para no
    # primeiro lance, em vez de materializar a lista inteira.
    legal_probe = next(
        board.generate_legal_moves(),
        None,
    )
    SEARCH_STATS["movegen_nodes"] += 1

    if legal_probe is None:
        return 0

    if q_depth <= 0:
        return get_static_eval(
            board,
            node_key=node_key,
        )

    stand_pat = get_static_eval(
        board,
        node_key=node_key,
    )

    if stand_pat >= beta:
        return beta

    if stand_pat > alpha:
        alpha = stand_pat

    # 2.4: nos níveis internos gera diretamente capturas/promoções.
    # No topo acrescenta ainda quiet checks para preservar força tática.
    tactical_infos = order_tactical_move_infos(
        board,
        None,
        q_depth,
        board_key=node_key,
    )

    for info in tactical_infos:
        check_time(deadline)
        move = info.move

        # SEE pruning + delta pruning.
        if (
            info.is_capture
            and move.promotion is None
            and not info.gives_check
        ):
            if static_exchange_eval(
                board,
                move,
                board_key=node_key,
            ) < -80:
                continue

            if board.is_en_passant(move):
                captured_value = PIECE_VALUES[
                    chess.PAWN
                ]
            else:
                captured_piece = board.piece_at(
                    move.to_square
                )
                captured_value = (
                    PIECE_VALUES[
                        captured_piece.piece_type
                    ]
                    if captured_piece is not None
                    else 0
                )

            if (
                stand_pat
                + captured_value
                + DELTA_MARGIN
                < alpha
            ):
                continue

        child_key = push_search_move(
            board,
            move,
            node_key,
            track_repetition=track_repetition,
        )

        try:
            score = -quiescence(
                board,
                -beta,
                -alpha,
                deadline,
                q_depth - 1,
                ply + 1,
                node_key=child_key,
                track_repetition=track_repetition,
            )
        finally:
            pop_search_move(
                board,
                child_key,
                track_repetition=track_repetition,
            )

        if score >= beta:
            SEARCH_STATS["cutoffs"] += 1

            if info.is_capture:
                update_capture_history(
                    board,
                    move,
                    max(1, q_depth),
                    success=True,
                )

            return beta

        if score > alpha:
            alpha = score

    return alpha




def is_advanced_passed_pawn_move(board, move, moving_piece=None):
    if moving_piece is None:
        moving_piece = board.piece_at(move.from_square)

    if moving_piece is None or moving_piece.piece_type != chess.PAWN:
        return False

    rank = chess.square_rank(move.to_square)

    if moving_piece.color == chess.WHITE:
        if rank < 5:
            return False
    else:
        if rank > 2:
            return False

    return is_passed_pawn(
        board,
        move.to_square,
        moving_piece.color,
    )


def _count_legal_replies_up_to(board, limit):
    count = 0

    for _ in board.generate_legal_moves():
        count += 1
        if count > limit:
            break

    return count


def extract_pv(board, max_length=16):
    """Reconstrói a PV a partir dos best_move guardados na TT."""
    temp = board.copy(stack=True)
    pv = []
    seen = set()

    for _ in range(max_length):
        key = get_board_key(temp)

        if key in seen:
            break
        seen.add(key)

        entry = get_tt_entry(key)
        if entry is None or entry.best_move is None:
            break

        move = entry.best_move
        if move not in temp.legal_moves:
            break

        pv.append(move)
        temp.push(move)

        if temp.is_game_over(claim_draw=False):
            break

    return pv


def pv_prefix_is_stable(previous_pv, current_pv, length=PV_STABILITY_LENGTH):
    if not previous_pv or not current_pv:
        return False

    required = min(length, len(previous_pv), len(current_pv))

    if required < 2:
        return False

    return previous_pv[:required] == current_pv[:required]


def pv_is_tactical(board, pv, max_plies=TACTICAL_PV_SCAN_PLIES):
    if not pv:
        return False

    temp = board.copy(stack=True)

    for move in pv[:max_plies]:
        if move not in temp.legal_moves:
            break

        if temp.is_capture(move) or move.promotion is not None or temp.gives_check(move):
            return True

        moving_piece = temp.piece_at(move.from_square)
        if moving_piece is not None and moving_piece.piece_type not in (chess.PAWN, chess.KING):
            if temp.is_attacked_by(not moving_piece.color, move.to_square):
                return True

        temp.push(move)

    return False

def negamax(
    board,
    depth,
    alpha,
    beta,
    deadline=None,
    ply=0,
    allow_null=True,
    node_key=None,
    track_repetition=True,
    excluded_move=None,
    allow_singular=True,
):
    check_time(deadline)
    SEARCH_STATS["nodes"] += 1

    if node_key is None:
        node_key = get_board_key(board)

    if depth <= 0:
        return quiescence(
            board,
            alpha,
            beta,
            deadline,
            QUIESCENCE_DEPTH,
            ply,
            node_key=node_key,
            track_repetition=track_repetition,
        )

    in_check = board.is_check()
    legal_moves = list(board.legal_moves)
    SEARCH_STATS["movegen_nodes"] += 1

    terminal = terminal_from_legal_moves(
        board,
        legal_moves,
        ply,
        node_key,
        track_repetition=track_repetition,
        in_check=in_check,
    )

    if terminal is not None:
        return terminal

    alpha_original = alpha
    beta_original = beta
    is_exclusion_search = excluded_move is not None

    repetition_sensitive = (
        track_repetition
        and repetition_count(node_key) >= 2
    )

    tt_entry = None
    tt_score = None
    tt_move = None

    if not repetition_sensitive and not is_exclusion_search:
        tt_entry = get_tt_entry(node_key)
        tt_score, tt_move = probe_tt(
            board,
            depth,
            alpha,
            beta,
            key=node_key,
        )

    if tt_score is not None:
        return tt_score

    narrow_window = beta - alpha <= 1
    static_eval = None
    fast_eval = None

    # -------------------------------------------------
    # REVERSE FUTILITY PRUNING COM FAST-EVAL PREFILTER
    # -------------------------------------------------
    use_rfp = (
        not is_exclusion_search
        and depth in REVERSE_FUTILITY_MARGINS
        and narrow_window
        and not in_check
        and alpha > -MATE_SCORE + 1000
        and beta < MATE_SCORE - 1000
    )

    if use_rfp:
        fast_eval = get_fast_eval(board, node_key=node_key)
        margin = REVERSE_FUTILITY_MARGINS[depth]

        if fast_eval - margin - FAST_EVAL_SAFETY_MARGIN >= beta:
            static_eval = get_static_eval(board, node_key=node_key)

            if static_eval - margin >= beta:
                SEARCH_STATS["rfp_prunes"] += 1
                SEARCH_STATS["cutoffs"] += 1
                return static_eval

    # -------------------------------------------------
    # NULL MOVE ADAPTATIVO, mas só chama full eval quando
    # o material rápido diz que o cutoff é plausível.
    # -------------------------------------------------
    has_non_pawn_material = bool(
        board.pieces_mask(chess.KNIGHT, board.turn)
        | board.pieces_mask(chess.BISHOP, board.turn)
        | board.pieces_mask(chess.ROOK, board.turn)
        | board.pieces_mask(chess.QUEEN, board.turn)
    )

    null_candidate = (
        not is_exclusion_search
        and allow_null
        and depth >= 3
        and narrow_window
        and not in_check
        and has_non_pawn_material
        and beta < MATE_SCORE - 1000
        and beta > -MATE_SCORE + 1000
    )

    if null_candidate:
        if fast_eval is None:
            fast_eval = get_fast_eval(board, node_key=node_key)

        if fast_eval >= beta - FAST_EVAL_SAFETY_MARGIN:
            if static_eval is None:
                static_eval = get_static_eval(board, node_key=node_key)

            if static_eval >= beta:
                null_reduction = min(4, 2 + depth // 6)
                reduced_depth = max(0, depth - 1 - null_reduction)

                null_key = push_with_zobrist(board, chess.Move.null(), node_key)

                try:
                    null_score = -negamax(
                        board,
                        reduced_depth,
                        -beta,
                        -beta + 1,
                        deadline,
                        ply + 1,
                        allow_null=False,
                        node_key=null_key,
                        track_repetition=False,
                        allow_singular=False,
                    )
                finally:
                    board.pop()

                if null_score >= beta:
                    SEARCH_STATS["null_cutoffs"] += 1
                    SEARCH_STATS["cutoffs"] += 1
                    return beta

    # -------------------------------------------------
    # SINGULAR EXTENSION
    # Só tenta provar singularidade quando a TT tem uma entrada profunda
    # EXACT/LOWER. A pesquisa de alternativas exclui temporariamente o TT move.
    # -------------------------------------------------
    singular_move = None

    if (
        allow_singular
        and not is_exclusion_search
        and narrow_window
        and tt_entry is not None
        and tt_move is not None
        and depth >= SINGULAR_EXTENSION_MIN_DEPTH
        and tt_entry.depth >= depth - 2
        and tt_entry.flag in (TT_EXACT, TT_LOWER)
        and abs(tt_entry.score) < MATE_SCORE - 2000
    ):
        singular_beta = (
            tt_entry.score
            - SINGULAR_EXTENSION_MARGIN
            - 4 * depth
        )
        singular_depth = max(1, (depth - 1) // 2)

        alternative_score = negamax(
            board,
            singular_depth,
            singular_beta - 1,
            singular_beta,
            deadline,
            ply,
            allow_null=False,
            node_key=node_key,
            track_repetition=track_repetition,
            excluded_move=tt_move,
            allow_singular=False,
        )

        if alternative_score < singular_beta:
            singular_move = tt_move

    # -------------------------------------------------
    # PESQUISA NORMAL
    # -------------------------------------------------
    best_score = -INFINITY
    best_move = None

    move_infos = iter_ordered_move_infos(
        board,
        legal_moves,
        ply=ply,
        tt_move=tt_move,
        board_key=node_key,
    )

    killer_moves = KILLER_MOVES.get(ply, [])
    previous_move = board.peek() if board.move_stack else None
    quiet_moves_seen = 0
    searched_quiet_moves = []

    for move_index, info in enumerate(move_infos):
        check_time(deadline)

        move = info.move

        if excluded_move is not None and move == excluded_move:
            continue

        quiet_before_push = info.is_quiet
        gives_check = info.gives_check
        killer_move = move in killer_moves
        history_score = quiet_history_score(move, previous_move)
        moving_piece = board.piece_at(move.from_square)

        advanced_pawn_push = (
            moving_piece is not None
            and moving_piece.piece_type == chess.PAWN
            and (
                (moving_piece.color == chess.WHITE and chess.square_rank(move.to_square) >= 5)
                or (moving_piece.color == chess.BLACK and chess.square_rank(move.to_square) <= 2)
            )
        )

        if quiet_before_push:
            quiet_moves_seen += 1

        use_lmp = (
            not is_exclusion_search
            and depth in LMP_QUIET_LIMITS
            and narrow_window
            and quiet_before_push
            and quiet_moves_seen > LMP_QUIET_LIMITS[depth]
            and not gives_check
            and not in_check
            and move != tt_move
            and not killer_move
            and not advanced_pawn_push
            and history_score < 500
            and alpha > -MATE_SCORE + 1000
            and beta < MATE_SCORE - 1000
        )

        if use_lmp:
            SEARCH_STATS["lmp_prunes"] += 1
            continue

        use_futility = (
            not is_exclusion_search
            and depth in FUTILITY_MARGINS
            and move_index >= 1
            and quiet_before_push
            and not gives_check
            and not in_check
            and move != tt_move
            and not killer_move
            and not advanced_pawn_push
            and narrow_window
            and alpha > -MATE_SCORE + 1000
            and beta < MATE_SCORE - 1000
        )

        if use_futility:
            if fast_eval is None:
                fast_eval = get_fast_eval(board, node_key=node_key)

            # Fast eval serve apenas como pré-filtro. Se parecer claramente
            # abaixo de alpha, a avaliação completa confirma antes do prune.
            if (
                fast_eval
                + FUTILITY_MARGINS[depth]
                + FAST_EVAL_SAFETY_MARGIN
                <= alpha
            ):
                if static_eval is None:
                    static_eval = get_static_eval(board, node_key=node_key)

                if static_eval + FUTILITY_MARGINS[depth] <= alpha:
                    continue

        use_lmr = (
            depth >= 3
            and move_index >= 3
            and quiet_before_push
            and not gives_check
            and not in_check
            and move != tt_move
            and not killer_move
            and not advanced_pawn_push
            and move != singular_move
        )

        lmr_reduction = 1

        if depth >= 6 and move_index >= 6:
            lmr_reduction += 1

        if depth >= 10 and move_index >= 12:
            lmr_reduction += 1

        if history_score >= 1000:
            lmr_reduction = max(1, lmr_reduction - 1)

        extension = 0

        is_recapture = (
            info.is_capture
            and previous_move is not None
            and move.to_square == previous_move.to_square
            and depth <= RECAPTURE_EXTENSION_MAX_DEPTH
        )

        if is_recapture:
            recapture_see = static_exchange_eval(board, move, board_key=node_key)

            if recapture_see >= RECAPTURE_EXTENSION_SEE_MIN:
                extension = 1
                SEARCH_STATS["recapture_extensions"] += 1

        if move == singular_move:
            extension = 1
            SEARCH_STATS["singular_extensions"] += 1

        passed_extension_candidate = (
            depth <= PASSED_PAWN_EXTENSION_MAX_DEPTH
            and is_advanced_passed_pawn_move(board, move, moving_piece)
        )

        if passed_extension_candidate and extension == 0:
            extension = 1
            SEARCH_STATS["passed_pawn_extensions"] += 1

        child_key = push_search_move(
            board,
            move,
            node_key,
            track_repetition=track_repetition,
        )

        try:
            # Check extension apenas para checks muito forçados (<=2 respostas).
            if (
                extension == 0
                and gives_check
                and depth <= SELECTIVE_CHECK_EXTENSION_MAX_DEPTH
            ):
                replies = _count_legal_replies_up_to(
                    board,
                    SELECTIVE_CHECK_MAX_REPLIES,
                )

                if replies <= SELECTIVE_CHECK_MAX_REPLIES:
                    extension = 1
                    SEARCH_STATS["check_extensions"] += 1

            child_depth = max(0, depth - 1 + extension)

            if extension:
                use_lmr = False

            if quiet_before_push:
                searched_quiet_moves.append(move)

            if move_index == 0:
                score = -negamax(
                    board,
                    child_depth,
                    -beta,
                    -alpha,
                    deadline,
                    ply + 1,
                    node_key=child_key,
                    track_repetition=track_repetition,
                )

            else:
                if use_lmr:
                    reduced_depth = max(1, child_depth - lmr_reduction)

                    score = -negamax(
                        board,
                        reduced_depth,
                        -alpha - 1,
                        -alpha,
                        deadline,
                        ply + 1,
                        node_key=child_key,
                        track_repetition=track_repetition,
                    )

                    if score > alpha:
                        score = -negamax(
                            board,
                            child_depth,
                            -alpha - 1,
                            -alpha,
                            deadline,
                            ply + 1,
                            node_key=child_key,
                            track_repetition=track_repetition,
                        )

                else:
                    score = -negamax(
                        board,
                        child_depth,
                        -alpha - 1,
                        -alpha,
                        deadline,
                        ply + 1,
                        node_key=child_key,
                        track_repetition=track_repetition,
                    )

                if score > alpha and score < beta:
                    score = -negamax(
                        board,
                        child_depth,
                        -beta,
                        -alpha,
                        deadline,
                        ply + 1,
                        node_key=child_key,
                        track_repetition=track_repetition,
                    )

        finally:
            pop_search_move(
                board,
                child_key,
                track_repetition=track_repetition,
            )

        if score > best_score:
            best_score = score
            best_move = move

        if score > alpha:
            alpha = score

        if alpha >= beta:
            SEARCH_STATS["cutoffs"] += 1

            if quiet_before_push:
                add_killer_move(ply, move)
                update_history(move, depth, previous_move=previous_move)

                for earlier_quiet in searched_quiet_moves:
                    if earlier_quiet != move:
                        penalize_history(
                            earlier_quiet,
                            depth,
                            previous_move=previous_move,
                        )
            elif info.is_capture:
                update_capture_history(board, move, depth, success=True)

            break

    if best_score == -INFINITY:
        if is_exclusion_search:
            return -INFINITY + ply

        best_score = get_static_eval(board, node_key=node_key)

    if best_score <= alpha_original:
        flag = TT_UPPER
    elif best_score >= beta_original:
        flag = TT_LOWER
    else:
        flag = TT_EXACT

    if not repetition_sensitive and not is_exclusion_search:
        store_tt(
            board,
            depth,
            best_score,
            flag,
            best_move,
            key=node_key,
        )

    return best_score





def calculate_root_ai_scores(board, moves):
    """Avalia todos os filhos da raiz num único batch PyTorch."""
    root_color = board.turn
    child_boards = []

    for move in moves:
        board.push(move)

        try:
            child_boards.append(board.copy(stack=False))
        finally:
            board.pop()

    raw_scores = evaluate_ai_batch_cp(
        child_boards,
        scale=AI_SCALE,
    )

    ai_scores = {}

    for move, ai_score in zip(moves, raw_scores):
        if root_color == chess.BLACK:
            ai_score = -ai_score

        ai_scores[move] = ai_score

    return ai_scores


def order_root_moves_with_ai(
    board,
    legal_moves,
    tt_move=None,
    board_key=None,
):
    """
    Move ordering da raiz com AI, reutilizando a metadata do move picker.
    """
    if board_key is None:
        board_key = get_board_key(board)

    root_cache_key = (
        board_key,
        board.fullmove_number,
    )

    ai_scores = ROOT_AI_SCORE_CACHE.get(root_cache_key)

    moves = list(legal_moves)

    if ai_scores is None:
        ai_scores = calculate_root_ai_scores(
            board,
            moves,
        )

        if len(ROOT_AI_SCORE_CACHE) >= MAX_ROOT_AI_CACHE_ENTRIES:
            ROOT_AI_SCORE_CACHE.clear()

        ROOT_AI_SCORE_CACHE[root_cache_key] = ai_scores

    infos = order_moves_from_list(
        board,
        moves,
        ply=0,
        tt_move=tt_move,
        board_key=board_key,
    )

    infos.sort(
        key=lambda info: (
            info.order_score
            + int(
                AI_ROOT_ORDER_WEIGHT
                * ai_scores.get(info.move, 0)
            )
        ),
        reverse=True,
    )

    return infos, ai_scores



def _captured_piece_value(board, move):
    if not board.is_capture(move):
        return 0

    if board.is_en_passant(move):
        return PIECE_VALUES[chess.PAWN]

    victim = board.piece_at(move.to_square)

    if victim is None:
        return 0

    return PIECE_VALUES.get(
        victim.piece_type,
        0,
    )


def root_move_material_risk(
    board,
    move,
    root_key,
):
    """Compatibilidade: indica se o lance merece verificação material."""
    extension, moving_value = root_move_verification_extension(
        board,
        move,
        root_key,
    )
    return extension > 0, moving_value


def root_move_verification_extension(board, move, root_key):
    """
    Profundidade adicional adaptativa para um sacrifício aparente na raiz.

    Não rejeita o lance. Apenas exige mais prova quando ele se torna candidato
    a melhor lance.
    """
    moving_piece = board.piece_at(move.from_square)

    if moving_piece is None:
        return 0, 0

    moving_value = PIECE_VALUES.get(moving_piece.piece_type, 0)

    if moving_piece.piece_type in (chess.PAWN, chess.KING):
        return 0, moving_value

    extension = 0

    if board.is_capture(move):
        victim_value = _captured_piece_value(board, move)
        material_gap = moving_value - victim_value
        see_score = static_exchange_eval(board, move, board_key=root_key)

        if moving_piece.piece_type == chess.QUEEN and material_gap >= 250:
            extension = max(extension, ROOT_VERIFY_QUEEN_PLIES)
        elif moving_piece.piece_type == chess.ROOK and material_gap >= 120:
            extension = max(extension, ROOT_VERIFY_ROOK_MINOR_PLIES)
        elif moving_piece.piece_type in (chess.KNIGHT, chess.BISHOP) and material_gap >= 150:
            extension = max(extension, ROOT_VERIFY_MINOR_PAWN_PLIES)
        elif material_gap >= ROOT_TACTICAL_VERIFY_MARGIN:
            extension = max(extension, 1)

        if see_score <= -300:
            extension = max(extension, 3)
        elif see_score <= -140:
            extension = max(extension, 2)
        elif see_score <= -70:
            extension = max(extension, 1)

    return min(ROOT_VERIFY_MAX_PLIES, extension), moving_value



def root_child_material_exposure_gain(
    board,
    move,
    child_key,
    moving_value,
):
    """Ganho SEE imediato que o adversário pode obter sobre a peça movida."""
    if moving_value < PIECE_VALUES[chess.KNIGHT]:
        return 0

    moved_piece = board.piece_at(move.to_square)

    if (
        moved_piece is None
        or moved_piece.color == board.turn
        or moved_piece.piece_type == chess.KING
    ):
        return 0

    recapture = least_valuable_legal_attacker(board, move.to_square)

    if recapture is None:
        return 0

    return max(
        0,
        static_exchange_eval(
            board,
            recapture,
            board_key=child_key,
        ),
    )


def root_child_exposes_moved_piece(
    board,
    move,
    child_key,
    moving_value,
):
    return (
        root_child_material_exposure_gain(
            board,
            move,
            child_key,
            moving_value,
        )
        >= ROOT_TACTICAL_VERIFY_MARGIN
    )



def search_root(
    board,
    depth,
    deadline=None,
    preferred_move=None,
    alpha=-INFINITY,
    beta=INFINITY,
):
    best_move = None
    best_score = -INFINITY

    original_alpha = alpha
    root_key = get_board_key(board)
    in_check = board.is_check()
    legal_moves = list(board.legal_moves)
    SEARCH_STATS["movegen_nodes"] += 1

    terminal = terminal_from_legal_moves(
        board,
        legal_moves,
        0,
        root_key,
        track_repetition=True,
        in_check=in_check,
    )

    if terminal is not None:
        return None, terminal

    tt_score, tt_move = probe_tt(
        board,
        depth,
        alpha,
        beta,
        key=root_key,
    )

    if preferred_move is not None:
        tt_move = preferred_move

    if USE_AI:
        move_infos, _ = order_root_moves_with_ai(
            board,
            legal_moves,
            tt_move=tt_move,
            board_key=root_key,
        )
    else:
        move_infos = order_moves_from_list(
            board,
            legal_moves,
            ply=0,
            tt_move=tt_move,
            board_key=root_key,
        )

    for move_index, info in enumerate(move_infos):
        check_time(deadline)
        move = info.move

        verification_extension, moving_value = root_move_verification_extension(
            board,
            move,
            root_key,
        )

        child_key = push_search_move(
            board,
            move,
            root_key,
            track_repetition=True,
        )

        exposure_gain = root_child_material_exposure_gain(
            board,
            move,
            child_key,
            moving_value,
        )

        if exposure_gain >= 300:
            verification_extension = max(verification_extension, 3)
        elif exposure_gain >= ROOT_TACTICAL_VERIFY_MARGIN:
            verification_extension = max(verification_extension, 2)

        verification_extension = min(
            ROOT_VERIFY_MAX_PLIES,
            verification_extension,
        )

        base_child_depth = max(0, depth - 1)

        try:
            if repetition_count(child_key) >= 3:
                SEARCH_STATS["repetition_draws"] += 1
                score = 0

            elif move_index == 0:
                score = -negamax(
                    board,
                    base_child_depth,
                    -beta,
                    -alpha,
                    deadline,
                    1,
                    node_key=child_key,
                    track_repetition=True,
                )

            else:
                score = -negamax(
                    board,
                    base_child_depth,
                    -alpha - 1,
                    -alpha,
                    deadline,
                    1,
                    node_key=child_key,
                    track_repetition=True,
                )

                if score > alpha and score < beta:
                    score = -negamax(
                        board,
                        base_child_depth,
                        -beta,
                        -alpha,
                        deadline,
                        1,
                        node_key=child_key,
                        track_repetition=True,
                    )

            # Só paga o custo extra se o sacrifício parece realmente tornar-se
            # o novo candidato principal (ou provocar fail-high).
            needs_deep_verification = (
                verification_extension > 0
                and score > best_score
            )

            if needs_deep_verification:
                SEARCH_STATS["root_verifications"] += 1
                SEARCH_STATS["root_verification_plies"] += verification_extension

                verified_depth = base_child_depth + verification_extension
                score = -negamax(
                    board,
                    verified_depth,
                    -beta,
                    -alpha,
                    deadline,
                    1,
                    node_key=child_key,
                    track_repetition=True,
                    allow_singular=True,
                )

        finally:
            pop_search_move(
                board,
                child_key,
                track_repetition=True,
            )

        if score > best_score:
            best_score = score
            best_move = move

        if score > alpha:
            alpha = score

        if alpha >= beta:
            break

    if best_score <= original_alpha:
        flag = TT_UPPER
    elif best_score >= beta:
        flag = TT_LOWER
    else:
        flag = TT_EXACT

    store_tt(
        board,
        depth,
        best_score,
        flag,
        best_move,
        key=root_key,
    )

    return best_move, best_score



def find_best_move(board, depth=3):
    reset_search_stats()
    _new_tt_generation()
    prepare_search_repetitions(board)
    return search_root(board, depth)

def find_best_move_timed(board, time_limit=1.0, max_depth=64):
    reset_search_stats()
    _new_tt_generation()
    prepare_search_repetitions(board)

    moves = list(board.legal_moves)

    if not moves:
        return None, 0, 0

    best_move = moves[0]
    best_score = 0
    completed_depth = 0

    last_completed_move = None
    last_completed_score = None
    last_completed_pv = []
    stable_iterations = 0

    time_limit = max(0.03, time_limit)
    deadline = time.perf_counter() + time_limit
    ASPIRATION_WINDOW = 50

    for depth in range(1, max_depth + 1):
        try:
            if depth == 1:
                move, score = search_root(
                    board,
                    depth,
                    deadline,
                    preferred_move=best_move,
                    alpha=-INFINITY,
                    beta=INFINITY,
                )
            else:
                window = ASPIRATION_WINDOW

                while True:
                    alpha = best_score - window
                    beta = best_score + window

                    move, score = search_root(
                        board,
                        depth,
                        deadline,
                        preferred_move=best_move,
                        alpha=alpha,
                        beta=beta,
                    )

                    if score <= alpha:
                        window *= 2
                    elif score >= beta:
                        window *= 2
                    else:
                        break

                    if window >= 2000:
                        move, score = search_root(
                            board,
                            depth,
                            deadline,
                            preferred_move=best_move,
                            alpha=-INFINITY,
                            beta=INFINITY,
                        )
                        break

        except SearchTimeout:
            break

        if move is not None:
            best_move = move
            best_score = score
            completed_depth = depth

            current_pv = extract_pv(
                board,
                max_length=max(PV_STABILITY_LENGTH + 2, min(16, depth + 2)),
            )

            same_move = (
                last_completed_move is not None
                and move == last_completed_move
            )

            stable_score = (
                last_completed_score is not None
                and abs(score - last_completed_score) <= 25
            )

            stable_pv = pv_prefix_is_stable(
                last_completed_pv,
                current_pv,
                length=PV_STABILITY_LENGTH,
            )

            tactical_line = pv_is_tactical(
                board,
                current_pv,
                max_plies=TACTICAL_PV_SCAN_PLIES,
            )

            would_be_stable = same_move and stable_score and stable_pv

            if would_be_stable and tactical_line:
                SEARCH_STATS["tactical_early_stop_blocks"] += 1
                stable_iterations = 0
            elif would_be_stable:
                stable_iterations += 1
            else:
                stable_iterations = 0

            last_completed_move = move
            last_completed_score = score
            last_completed_pv = current_pv

            if (
                depth >= EARLY_STOP_MIN_DEPTH
                and stable_iterations >= EARLY_STOP_STABLE_ITERS
            ):
                break

        if time.perf_counter() >= deadline:
            break

    return best_move, best_score, completed_depth

