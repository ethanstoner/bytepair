"""BPE training loops.

Both trainers take chunks as a {tuple_of_byte_ids: frequency} mapping and return
the ordered list of learned merges. `naive` recounts every pair on every step,
the way the lecture does it. `incremental` keeps a running pair count and an
index from pair to the chunks that contain it, so each merge only touches the
chunks it actually changes. Both break ties the same way (see core.best_pair),
so they learn identical merges.
"""

import heapq
from collections import Counter

from .core import best_pair, pair_counts, replace_pair


def naive(chunks, num_merges, first_id=256):
    words = [(list(ids), freq) for ids, freq in chunks.items()]
    merges = []
    for step in range(num_merges):
        counts = {}
        for ids, freq in words:
            pair_counts(ids, counts, freq)
        if not counts:
            break
        pair = best_pair(counts)
        new_id = first_id + step
        words = [(replace_pair(ids, pair, new_id), freq) for ids, freq in words]
        merges.append(pair)
    return merges


def incremental(chunks, num_merges, first_id=256):
    words = [list(ids) for ids in chunks]
    freqs = list(chunks.values())
    counts = {}
    where = {}
    for idx, ids in enumerate(words):
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + freqs[idx]
            where.setdefault(pair, set()).add(idx)

    # Max-heap via negated counts; tuple order breaks ties on the smallest pair.
    # Entries go stale when a count changes and are skipped when popped.
    heap = [(-c, pair) for pair, c in counts.items()]
    heapq.heapify(heap)

    merges = []
    for step in range(num_merges):
        pair = None
        while heap:
            neg, cand = heapq.heappop(heap)
            if counts.get(cand) == -neg:
                pair = cand
                break
        if pair is None:
            break
        new_id = first_id + step
        touched = set()
        for idx in where.pop(pair, ()):
            old = words[idx]
            new = replace_pair(old, pair, new_id)
            if len(new) == len(old):
                continue  # index entry was stale
            freq = freqs[idx]
            # Only pairs whose count changed inside this word need updating.
            before = Counter(zip(old, old[1:]))
            after = Counter(zip(new, new[1:]))
            for p, n in before.items():
                delta = after.get(p, 0) - n
                if delta:
                    counts[p] += delta * freq
                    touched.add(p)
            for p, n in after.items():
                if p not in before:
                    counts[p] = counts.get(p, 0) + n * freq
                    where.setdefault(p, set()).add(idx)
                    touched.add(p)
            words[idx] = new
        for p in touched:
            c = counts[p]
            if c > 0:
                heapq.heappush(heap, (-c, p))
            else:
                del counts[p]
        merges.append(pair)
    return merges


TRAINERS = {"naive": naive, "incremental": incremental}
