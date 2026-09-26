from bytepair.core import best_pair, pair_counts, render_token, replace_pair


def test_pair_counts():
    assert pair_counts([1, 2, 3, 1, 2]) == {(1, 2): 2, (2, 3): 1, (3, 1): 1}


def test_pair_counts_accumulates_with_weight():
    counts = pair_counts([1, 2], weight=3)
    pair_counts([1, 2, 1], counts)
    assert counts == {(1, 2): 4, (2, 1): 1}


def test_replace_pair_non_overlapping():
    assert replace_pair([1, 1, 1, 2, 1, 1], (1, 1), 9) == [9, 1, 2, 9]
    assert replace_pair([], (1, 1), 9) == []
    assert replace_pair([1], (1, 1), 9) == [1]


def test_best_pair_breaks_ties_on_smallest_pair():
    assert best_pair({(5, 1): 3, (2, 9): 3, (2, 4): 3, (0, 0): 1}) == (2, 4)


def test_render_token_escapes_control_chars():
    assert render_token(b"a\nb") == "a\\u000ab"
    assert render_token("héllo".encode()) == "héllo"
