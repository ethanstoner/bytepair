"""Small building blocks shared by every tokenizer."""

import unicodedata


def pair_counts(ids, counts=None, weight=1):
    """Count adjacent pairs in `ids`, adding into `counts` if one is given."""
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + weight
    return counts


def replace_pair(ids, pair, new_id):
    """Return a copy of `ids` with every non-overlapping `pair` swapped for `new_id`."""
    a, b = pair
    out = []
    i = 0
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def best_pair(counts):
    """Most frequent pair; ties go to the smallest pair so every trainer agrees."""
    return max(counts.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))[0]


def render_token(raw):
    """Readable form of a token's bytes, with control characters escaped."""
    text = raw.decode("utf-8", errors="replace")
    return "".join(
        f"\\u{ord(ch):04x}" if unicodedata.category(ch)[0] == "C" else ch
        for ch in text
    )
