//! Applying learned merges to one pre-tokenized chunk.

use rustc_hash::FxHashMap;

/// Pair -> merged id. A merged id is also its rank: lower ids were learned earlier.
pub type Ranks = FxHashMap<u64, u32>;

#[inline]
pub fn key(a: u32, b: u32) -> u64 {
    (a as u64) << 32 | b as u64
}

/// Merge `ids` in place with the rank-array algorithm: keep the rank of every adjacent
/// pair, merge the leftmost lowest-ranked one, re-rank only its two neighbours.
///
/// Any pair containing a freshly merged id ranks after it, so merging one occurrence at a
/// time left to right gives the same result as replacing every occurrence at once.
pub fn merge(ids: &mut Vec<u32>, ranks: &Ranks) {
    if ids.len() < 2 {
        return;
    }
    let get = |a: u32, b: u32| ranks.get(&key(a, b)).copied().unwrap_or(u32::MAX);
    let mut pair_ranks: Vec<u32> = ids.windows(2).map(|w| get(w[0], w[1])).collect();
    loop {
        let mut best = u32::MAX;
        let mut at = 0;
        for (i, &r) in pair_ranks.iter().enumerate() {
            if r < best {
                best = r;
                at = i;
            }
        }
        if best == u32::MAX {
            break;
        }
        ids[at] = best;
        ids.remove(at + 1);
        pair_ranks.remove(at);
        if at > 0 {
            pair_ranks[at - 1] = get(ids[at - 1], best);
        }
        if at < pair_ranks.len() {
            pair_ranks[at] = get(best, ids[at + 1]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ranks(merges: &[(u32, u32)]) -> Ranks {
        merges.iter().enumerate().map(|(i, &(a, b))| (key(a, b), 256 + i as u32)).collect()
    }

    #[test]
    fn wikipedia_example() {
        // The merges the Python trainer learns on "aaabdaaabac": aa, ab, then aa+ab.
        let r = ranks(&[(97, 97), (97, 98), (256, 257)]);
        let mut ids: Vec<u32> = b"aaabdaaabac".iter().map(|&b| b as u32).collect();
        merge(&mut ids, &r);
        assert_eq!(ids, [258, 100, 258, 97, 99]);
    }

    #[test]
    fn overlapping_runs_merge_left_to_right() {
        let r = ranks(&[(97, 97)]);
        let mut ids: Vec<u32> = b"aaaaaaa".iter().map(|&b| b as u32).collect();
        merge(&mut ids, &r);
        assert_eq!(ids, [256, 256, 256, 97]);
    }

    #[test]
    fn nothing_to_merge() {
        let mut ids = vec![1, 2, 3];
        merge(&mut ids, &ranks(&[(9, 9)]));
        assert_eq!(ids, [1, 2, 3]);
        let mut one = vec![5];
        merge(&mut one, &ranks(&[]));
        assert_eq!(one, [5]);
    }
}
