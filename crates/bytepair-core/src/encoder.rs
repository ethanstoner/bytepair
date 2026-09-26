//! Text <-> token ids: special tokens, pre-tokenization, cached per-chunk BPE, decoding.

use std::fmt;
use std::sync::Mutex;

use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::bpe::{self, key, Ranks};
use crate::pretokenize::{RegexError, Splitter};

/// Distinct chunks remembered by the shared cache (same limit as the Python reference).
pub const CACHE_LIMIT: usize = 1_000_000;

type Cache = FxHashMap<Box<str>, Vec<u32>>;

#[derive(Debug)]
pub enum Error {
    Regex(RegexError),
    DisallowedSpecial(String),
    UnknownSpecial(String),
    UnknownId(u32),
    Invalid(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Regex(e) => write!(f, "{e}"),
            Error::DisallowedSpecial(s) => write!(f, "text contains special token '{s}'"),
            Error::UnknownSpecial(s) => write!(f, "unknown special token '{s}'"),
            Error::UnknownId(i) => write!(f, "unknown token id {i}"),
            Error::Invalid(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<RegexError> for Error {
    fn from(e: RegexError) -> Self {
        Error::Regex(e)
    }
}

/// Which special tokens `encode` treats as specials.
pub enum Allowed<'a> {
    All,
    /// Special strings are encoded as ordinary text.
    None,
    /// Error if any special string appears (tiktoken's default).
    NoneRaise,
    Set(&'a [String]),
}

pub struct Encoder {
    ranks: Ranks,
    byte_rank: [u32; 256],
    vocab: Vec<Vec<u8>>, // empty = no token with this id
    specials: Vec<(String, u32)>, // insertion order = alternation order
    special_ids: FxHashMap<u32, String>,
    splitter: Splitter,
    cache: Mutex<Cache>,
}

impl Encoder {
    /// A trained tokenizer: merge `i` creates id `256 + i`; bytes map to themselves.
    pub fn from_merges(
        merges: &[(u32, u32)],
        pattern: Option<&str>,
        specials: Vec<(String, u32)>,
    ) -> Result<Self, Error> {
        let mut vocab: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
        let mut ranks = Ranks::default();
        for (i, &(a, b)) in merges.iter().enumerate() {
            let id = 256 + i as u32;
            let (a_bytes, b_bytes) = match (vocab.get(a as usize), vocab.get(b as usize)) {
                (Some(x), Some(y)) if !x.is_empty() && !y.is_empty() => (x, y),
                _ => return Err(Error::Invalid(format!("merge {i} uses an undefined id"))),
            };
            let joined = [a_bytes.as_slice(), b_bytes.as_slice()].concat();
            vocab.push(joined);
            ranks.insert(key(a, b), id);
        }
        let byte_rank = std::array::from_fn(|i| i as u32);
        Self::build(ranks, byte_rank, vocab, pattern, specials)
    }

    /// An OpenAI encoding from tiktoken's `{token bytes: rank}` table.
    ///
    /// Each multi-byte token was formed by merging two lower-ranked tokens. Running BPE on
    /// it with merges capped below its own rank stops at exactly those two parents.
    pub fn from_ranks(
        token_ranks: Vec<(Vec<u8>, u32)>,
        pattern: Option<&str>,
        specials: Vec<(String, u32)>,
    ) -> Result<Self, Error> {
        let by_bytes: FxHashMap<&[u8], u32> =
            token_ranks.iter().map(|(t, r)| (t.as_slice(), *r)).collect();
        let mut byte_rank = [u32::MAX; 256];
        for (t, r) in &token_ranks {
            if t.len() == 1 {
                byte_rank[t[0] as usize] = *r;
            }
        }
        if byte_rank.contains(&u32::MAX) {
            return Err(Error::Invalid("ranks do not cover all 256 single bytes".into()));
        }
        let pairs: Result<Vec<(u64, u32)>, Error> = token_ranks
            .par_iter()
            .filter(|(t, _)| t.len() > 1)
            .map(|(t, r)| {
                let parts = capped_bpe(&by_bytes, t, *r);
                match parts.as_slice() {
                    [a, b] => Ok((key(by_bytes[a], by_bytes[b]), *r)),
                    _ => Err(Error::Invalid(format!("token {t:?} has no two-part parent"))),
                }
            })
            .collect();
        let ranks: Ranks = pairs?.into_iter().collect();
        let size = token_ranks.iter().map(|(_, r)| *r as usize + 1).max().unwrap_or(0);
        let mut vocab = vec![Vec::new(); size];
        for (t, r) in token_ranks {
            vocab[r as usize] = t;
        }
        Self::build(ranks, byte_rank, vocab, pattern, specials)
    }

    fn build(
        ranks: Ranks,
        byte_rank: [u32; 256],
        vocab: Vec<Vec<u8>>,
        pattern: Option<&str>,
        specials: Vec<(String, u32)>,
    ) -> Result<Self, Error> {
        let special_ids = specials.iter().map(|(s, i)| (*i, s.clone())).collect();
        Ok(Encoder {
            ranks,
            byte_rank,
            vocab,
            specials,
            special_ids,
            splitter: Splitter::for_pattern(pattern)?,
            cache: Mutex::new(Cache::default()),
        })
    }

    pub fn clear_cache(&self) {
        self.cache.lock().unwrap().clear();
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.iter().filter(|t| !t.is_empty()).count() + self.specials.len()
    }

    fn encode_chunk(&self, chunk: &str) -> Vec<u32> {
        let mut ids: Vec<u32> = chunk.bytes().map(|b| self.byte_rank[b as usize]).collect();
        bpe::merge(&mut ids, &self.ranks);
        ids
    }

    fn encode_with(&self, cache: &mut Cache, text: &str, out: &mut Vec<u32>) -> Result<(), Error> {
        self.splitter.for_each(text, |chunk| {
            if let Some(ids) = cache.get(chunk) {
                out.extend_from_slice(ids);
            } else {
                let ids = self.encode_chunk(chunk);
                out.extend_from_slice(&ids);
                if cache.len() < CACHE_LIMIT {
                    cache.insert(chunk.into(), ids);
                }
            }
        })?;
        Ok(())
    }

    /// Encode, treating special-token strings as plain text. Uses the shared cache.
    pub fn encode_ordinary(&self, text: &str) -> Result<Vec<u32>, Error> {
        let mut out = Vec::with_capacity(text.len() / 3);
        let mut cache = self.cache.lock().unwrap();
        self.encode_with(&mut cache, text, &mut out)?;
        Ok(out)
    }

    pub fn encode(&self, text: &str, allowed: Allowed) -> Result<Vec<u32>, Error> {
        let active: Vec<&(String, u32)> = match allowed {
            Allowed::All => self.specials.iter().collect(),
            Allowed::None => Vec::new(),
            Allowed::NoneRaise => {
                if let Some((s, _)) = self.specials.iter().find(|(s, _)| text.contains(s.as_str())) {
                    return Err(Error::DisallowedSpecial(s.clone()));
                }
                Vec::new()
            }
            Allowed::Set(names) => {
                if let Some(bad) = names.iter().find(|n| !self.specials.iter().any(|(s, _)| s == *n)) {
                    return Err(Error::UnknownSpecial(bad.clone()));
                }
                self.specials.iter().filter(|(s, _)| names.contains(s)).collect()
            }
        };
        if active.is_empty() {
            return self.encode_ordinary(text);
        }

        let mut out = Vec::with_capacity(text.len() / 3);
        let mut cache = self.cache.lock().unwrap();
        let mut start = 0;
        // Leftmost-first alternation, like regex.split("(s1|s2|...)"): the earliest match
        // wins, and at a tie the special listed first wins.
        while let Some((at, s, id)) = active
            .iter()
            .filter_map(|(s, id)| text[start..].find(s.as_str()).map(|p| (start + p, s, *id)))
            .min_by_key(|&(p, _, _)| p)
        {
            self.encode_with(&mut cache, &text[start..at], &mut out)?;
            out.push(id);
            start = at + s.len();
        }
        self.encode_with(&mut cache, &text[start..], &mut out)?;
        Ok(out)
    }

    /// Encode many texts in parallel. Each worker keeps its own cache for this call, so
    /// threads never contend on the shared one.
    pub fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<u32>>, Error> {
        texts
            .par_iter()
            .map_init(Cache::default, |cache, text| {
                let mut out = Vec::with_capacity(text.len() / 3);
                self.encode_with(cache, text, &mut out)?;
                Ok(out)
            })
            .collect()
    }

    pub fn decode_bytes(&self, ids: &[u32]) -> Result<Vec<u8>, Error> {
        let mut out = Vec::with_capacity(ids.len() * 4);
        for &i in ids {
            match self.vocab.get(i as usize) {
                Some(t) if !t.is_empty() => out.extend_from_slice(t),
                _ => match self.special_ids.get(&i) {
                    Some(s) => out.extend_from_slice(s.as_bytes()),
                    None => return Err(Error::UnknownId(i)),
                },
            }
        }
        Ok(out)
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String, Error> {
        Ok(String::from_utf8_lossy(&self.decode_bytes(ids)?).into_owned())
    }
}

/// Split `token` with BPE using only merges ranked below `max_rank`.
fn capped_bpe<'t>(by_bytes: &FxHashMap<&[u8], u32>, token: &'t [u8], max_rank: u32) -> Vec<&'t [u8]> {
    let mut parts: Vec<&[u8]> = token.chunks(1).collect();
    loop {
        let mut best: Option<(usize, u32)> = None;
        for i in 0..parts.len().saturating_sub(1) {
            let joined_len = parts[i].len() + parts[i + 1].len();
            let start = parts[i].as_ptr() as usize - token.as_ptr() as usize;
            let joined = &token[start..start + joined_len];
            if let Some(&r) = by_bytes.get(joined) {
                if r < max_rank && best.is_none_or(|(_, b)| r < b) {
                    best = Some((i, r));
                }
            }
        }
        let Some((i, _)) = best else { return parts };
        let start = parts[i].as_ptr() as usize - token.as_ptr() as usize;
        parts[i] = &token[start..start + parts[i].len() + parts[i + 1].len()];
        parts.remove(i + 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trained() -> Encoder {
        Encoder::from_merges(
            &[(97, 97), (97, 98), (256, 257)],
            None,
            vec![("<|a|>".into(), 1000), ("<|a|>b".into(), 1001)],
        )
        .unwrap()
    }

    #[test]
    fn round_trip_and_specials() {
        let enc = trained();
        let ids = enc.encode("aaabdaaabac", Allowed::None).unwrap();
        assert_eq!(ids, [258, 100, 258, 97, 99]);
        assert_eq!(enc.decode(&ids).unwrap(), "aaabdaaabac");
        // Leftmost-first: "<|a|>" is listed first, so it wins over the longer "<|a|>b".
        assert_eq!(enc.encode("x<|a|>b", Allowed::All).unwrap(), [120, 1000, 98]);
        assert!(matches!(enc.encode("<|a|>", Allowed::NoneRaise), Err(Error::DisallowedSpecial(_))));
        let bad = ["nope".to_string()];
        assert!(matches!(enc.encode("x", Allowed::Set(&bad)), Err(Error::UnknownSpecial(_))));
        assert!(matches!(enc.decode(&[5000]), Err(Error::UnknownId(5000))));
    }

    #[test]
    fn batch_matches_single() {
        let enc = trained();
        let texts = ["aaab", "", "baaac", "aaabdaaabac"];
        let batch = enc.encode_batch(&texts).unwrap();
        for (t, ids) in texts.iter().zip(batch) {
            assert_eq!(ids, enc.encode_ordinary(t).unwrap());
        }
    }
}
