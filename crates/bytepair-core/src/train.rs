//! Learning merges: the incremental trainer, ported from the Python reference.
//!
//! Chunks are deduplicated with their frequencies. Pair counts are kept up to date and
//! indexed by the chunks that contain them, so each merge only revisits chunks holding the
//! merged pair. The best pair comes off a max-heap whose stale entries are skipped when
//! popped. Ties go to the smallest pair, exactly as in Python, so both learn the same merges.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::bpe::key;
use crate::pretokenize::{RegexError, Splitter};

fn unkey(k: u64) -> (u32, u32) {
    ((k >> 32) as u32, k as u32)
}

/// Non-overlapping, left-to-right replacement of `pair` with `new_id`.
fn replace(ids: &[u32], a: u32, b: u32, new_id: u32) -> Vec<u32> {
    let mut out = Vec::with_capacity(ids.len());
    let mut i = 0;
    while i < ids.len() {
        if i + 1 < ids.len() && ids[i] == a && ids[i + 1] == b {
            out.push(new_id);
            i += 2;
        } else {
            out.push(ids[i]);
            i += 1;
        }
    }
    out
}

/// The chunk's pairs, overlapping (`aaa` yields `(a, a)` twice), sorted into `buf`.
fn sorted_pairs(ids: &[u32], buf: &mut Vec<u64>) {
    buf.clear();
    buf.extend(ids.windows(2).map(|w| key(w[0], w[1])));
    buf.sort_unstable();
}

/// Learn up to `num_merges` merges from `{chunk bytes: frequency}`. Merge `i` gets id `256 + i`.
pub fn train_chunks(chunks: &FxHashMap<Vec<u8>, u64>, num_merges: usize) -> Vec<(u32, u32)> {
    let mut words: Vec<Vec<u32>> = Vec::with_capacity(chunks.len());
    let mut freqs: Vec<i64> = Vec::with_capacity(chunks.len());
    for (bytes, &freq) in chunks {
        words.push(bytes.iter().map(|&b| b as u32).collect());
        freqs.push(freq as i64);
    }

    let mut counts: FxHashMap<u64, i64> = words
        .par_iter()
        .zip(freqs.par_iter())
        .fold(FxHashMap::default, |mut acc, (ids, &freq)| {
            for w in ids.windows(2) {
                *acc.entry(key(w[0], w[1])).or_insert(0) += freq;
            }
            acc
        })
        .reduce(FxHashMap::default, |mut a, b| {
            if a.len() < b.len() {
                return merge_into(b, a);
            }
            for (k, v) in b {
                *a.entry(k).or_insert(0) += v;
            }
            a
        });
    let mut index: FxHashMap<u64, FxHashSet<u32>> = FxHashMap::default();
    for (idx, ids) in words.iter().enumerate() {
        for w in ids.windows(2) {
            index.entry(key(w[0], w[1])).or_default().insert(idx as u32);
        }
    }

    let mut heap: BinaryHeap<(i64, Reverse<u64>)> =
        counts.iter().map(|(&k, &c)| (c, Reverse(k))).collect();
    let mut merges = Vec::with_capacity(num_merges);
    let mut touched: FxHashSet<u64> = FxHashSet::default();

    for step in 0..num_merges {
        let pair = loop {
            match heap.pop() {
                None => break None,
                Some((c, Reverse(k))) if counts.get(&k) == Some(&c) => break Some(k),
                Some(_) => continue, // stale
            }
        };
        let Some(pair) = pair else { break };
        let (a, b) = unkey(pair);
        let new_id = 256 + step as u32;
        touched.clear();
        let (mut before, mut after) = (Vec::new(), Vec::new());

        for idx in index.remove(&pair).unwrap_or_default() {
            let old = &words[idx as usize];
            let new = replace(old, a, b, new_id);
            if new.len() == old.len() {
                continue; // stale index entry
            }
            let freq = freqs[idx as usize];
            // Walk both sorted pair lists together; only pairs whose count changed matter.
            sorted_pairs(old, &mut before);
            sorted_pairs(&new, &mut after);
            let (mut i, mut j) = (0, 0);
            while i < before.len() || j < after.len() {
                let p = match (before.get(i), after.get(j)) {
                    (Some(&x), Some(&y)) => x.min(y),
                    (Some(&x), None) => x,
                    (None, Some(&y)) => y,
                    (None, None) => unreachable!(),
                };
                let (mut nb, mut na) = (0i64, 0i64);
                while before.get(i) == Some(&p) {
                    nb += 1;
                    i += 1;
                }
                while after.get(j) == Some(&p) {
                    na += 1;
                    j += 1;
                }
                if na != nb {
                    *counts.entry(p).or_insert(0) += (na - nb) * freq;
                    touched.insert(p);
                    if nb == 0 {
                        index.entry(p).or_default().insert(idx);
                    }
                }
            }
            words[idx as usize] = new;
        }
        for &p in &touched {
            let c = counts[&p];
            if c > 0 {
                heap.push((c, Reverse(p)));
            } else {
                counts.remove(&p);
            }
        }
        merges.push((a, b));
    }
    merges
}

fn merge_into(mut big: FxHashMap<u64, i64>, small: FxHashMap<u64, i64>) -> FxHashMap<u64, i64> {
    for (k, v) in small {
        *big.entry(k).or_insert(0) += v;
    }
    big
}

/// Split `text` with `pattern` (None = whole text), count chunks, and learn merges.
pub fn train_from_text(
    text: &str,
    pattern: Option<&str>,
    vocab_size: usize,
) -> Result<Vec<(u32, u32)>, RegexError> {
    let splitter = Splitter::for_pattern(pattern)?;
    let mut by_str: FxHashMap<&str, u64> = FxHashMap::default();
    splitter.for_each(text, |chunk| *by_str.entry(chunk).or_insert(0) += 1)?;
    let chunks: FxHashMap<Vec<u8>, u64> =
        by_str.into_iter().map(|(s, n)| (s.as_bytes().to_vec(), n)).collect();
    Ok(train_chunks(&chunks, vocab_size.saturating_sub(256)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// Recount everything every step: slow, obviously correct.
    fn naive(chunks: &FxHashMap<Vec<u8>, u64>, num_merges: usize) -> Vec<(u32, u32)> {
        let mut words: Vec<(Vec<u32>, i64)> =
            chunks.iter().map(|(b, &f)| (b.iter().map(|&x| x as u32).collect(), f as i64)).collect();
        let mut merges = Vec::new();
        for step in 0..num_merges {
            let mut counts: FxHashMap<u64, i64> = FxHashMap::default();
            for (ids, f) in &words {
                for w in ids.windows(2) {
                    *counts.entry(key(w[0], w[1])).or_insert(0) += f;
                }
            }
            let Some((&best, _)) = counts.iter().max_by_key(|&(&k, &c)| (c, Reverse(k))) else { break };
            let (a, b) = unkey(best);
            for (ids, _) in &mut words {
                *ids = replace(ids, a, b, 256 + step as u32);
            }
            merges.push((a, b));
        }
        merges
    }

    #[test]
    fn wikipedia_example() {
        let merges = train_from_text("aaabdaaabac", None, 259).unwrap();
        assert_eq!(merges, [(97, 97), (97, 98), (256, 257)]);
    }

    #[test]
    fn stops_when_nothing_left() {
        assert_eq!(train_from_text("ab", None, 1000).unwrap(), [(97, 98)]);
        assert!(train_from_text("", None, 1000).unwrap().is_empty());
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(500))]
        #[test]
        fn incremental_matches_naive(
            words in prop::collection::vec(("[abcd]{1,12}", 1u64..6), 1..40),
            n in 1usize..30,
        ) {
            let mut chunks: FxHashMap<Vec<u8>, u64> = FxHashMap::default();
            for (w, f) in words {
                *chunks.entry(w.into_bytes()).or_insert(0) += f;
            }
            prop_assert_eq!(train_chunks(&chunks, n), naive(&chunks, n));
        }
    }
}
