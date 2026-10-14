import chess
import chess.polyglot

import engine


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"{name:<44} {status} {detail}")
    return bool(condition)


def test_incremental_zobrist():
    board = chess.Board()
    key = engine.get_board_key(board)

    moves = [
        "e2e4", "c7c5", "g1f3", "d7d6",
        "d2d4", "c5d4", "f3d4", "g8f6",
        "b1c3", "a7a6", "c1e3", "e7e6",
        "f1e2", "f8e7", "e1g1", "e8g8",
    ]

    for text in moves:
        move = chess.Move.from_uci(text)
        if move not in board.legal_moves:
            return False, f"jogada ilegal: {text}"

        key = engine.push_with_zobrist(board, move, key)
        if key != chess.polyglot.zobrist_hash(board):
            return False, f"mismatch depois de {text}"

    return True, "sequência normal + roques"


def test_ep_zobrist():
    board = chess.Board()
    key = engine.get_board_key(board)

    for text in ["e2e4", "a7a6", "e4e5", "d7d5", "e5d6"]:
        move = chess.Move.from_uci(text)
        key = engine.push_with_zobrist(board, move, key)

        if key != chess.polyglot.zobrist_hash(board):
            return False, f"mismatch EP em {text}"

    return True, "en-passant"


def test_promotion_zobrist():
    board = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    key = engine.get_board_key(board)
    move = chess.Move.from_uci("a7a8q")
    key = engine.push_with_zobrist(board, move, key)

    return key == chess.polyglot.zobrist_hash(board), "promoção"


def test_minor_sacrifice_verification():
    board = chess.Board("4k3/8/4p3/3p4/8/2N5/8/4K3 w - - 0 1")
    move = chess.Move.from_uci("c3d5")
    key = engine.get_board_key(board)
    extension, _ = engine.root_move_verification_extension(board, move, key)

    return extension >= 2, f"extensão={extension}"


def test_queen_sacrifice_verification():
    board = chess.Board("4k3/8/8/3r4/8/8/3Q4/4K3 w - - 0 1")
    move = chess.Move.from_uci("d2d5")
    key = engine.get_board_key(board)
    extension, _ = engine.root_move_verification_extension(board, move, key)

    return extension >= 3, f"extensão={extension}"


def test_fixed_tt():
    engine.set_hash_size(1)
    engine.clear_engine_state()

    board = chess.Board()
    key = engine.get_board_key(board)
    move = chess.Move.from_uci("e2e4")

    engine.store_tt(
        board,
        depth=4,
        score=23,
        flag=engine.TT_EXACT,
        best_move=move,
        key=key,
    )

    score, probed_move = engine.probe_tt(
        board,
        depth=4,
        alpha=-100,
        beta=100,
        key=key,
    )

    entry = engine.get_tt_entry(key)
    stats = engine.get_search_stats()

    ok = (
        score == 23
        and probed_move == move
        and entry is not None
        and stats["tt_entries"] == 1
    )

    return ok, f"entries={stats['tt_entries']}"


def test_fast_eval():
    board = chess.Board("4k3/8/8/8/8/8/3Q4/4K3 w - - 0 1")
    key = engine.get_board_key(board)
    score = engine.get_fast_eval(board, key)
    return score > 700, f"fast_eval={score}"


def test_countermove_history():
    engine.clear_engine_state()
    previous = chess.Move.from_uci("e7e5")
    reply = chess.Move.from_uci("g1f3")

    engine.update_history(reply, depth=5, previous_move=previous)
    stored = engine.COUNTERMOVE_HEURISTIC.get(engine.history_key(previous))
    score = engine.quiet_history_score(reply, previous)

    return stored == reply and score > 0, f"history={score}"


def test_pv_stability():
    pv1 = [
        chess.Move.from_uci("e2e4"),
        chess.Move.from_uci("e7e5"),
        chess.Move.from_uci("g1f3"),
        chess.Move.from_uci("b8c6"),
    ]
    pv2 = list(pv1)

    return engine.pv_prefix_is_stable(pv1, pv2, 4), "4 plies iguais"


def main():
    print("Testes específicos Barromax 2.4 deep-search")
    print("=" * 76)

    tests = [
        ("Zobrist incremental", test_incremental_zobrist),
        ("Zobrist en-passant", test_ep_zobrist),
        ("Zobrist promoção", test_promotion_zobrist),
        ("Verificação sacrifício menor", test_minor_sacrifice_verification),
        ("Verificação sacrifício dama", test_queen_sacrifice_verification),
        ("TT fixa 2-way", test_fixed_tt),
        ("Fast eval", test_fast_eval),
        ("Countermove/continuation history", test_countermove_history),
        ("PV stability", test_pv_stability),
    ]

    results = []
    for name, function in tests:
        ok, detail = function()
        results.append(check(name, ok, detail))

    print("=" * 76)
    print(f"Resultado: {sum(results)}/{len(results)}")

    if all(results):
        print("Excelente. As otimizações deep-search da 2.4 passaram.")
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
