//! Pre-tokenization: cutting text into the chunks BPE runs on.
//!
//! The two OpenAI patterns get hand-written matchers; anything else goes through
//! fancy-regex. Both produce exactly what `regex.findall(pattern, text)` produces.

mod cl100k;
mod o200k;

use fancy_regex::Regex;

pub const CL100K_PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s";
pub const O200K_PATTERN: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

#[derive(Debug)]
pub enum Splitter {
    Cl100k,
    O200k,
    Regex(Box<Regex>),
    /// No pre-splitting: the whole text is one chunk.
    Whole,
}

impl Splitter {
    /// The hand-written matcher when `pattern` is exactly a known pattern, else a regex.
    pub fn for_pattern(pattern: Option<&str>) -> Result<Self, fancy_regex::Error> {
        Ok(match pattern {
            None => Splitter::Whole,
            Some(CL100K_PATTERN) => Splitter::Cl100k,
            Some(O200K_PATTERN) => Splitter::O200k,
            Some(p) => Splitter::regex(p)?,
        })
    }

    pub fn regex(pattern: &str) -> Result<Self, fancy_regex::Error> {
        Ok(Splitter::Regex(Box::new(Regex::new(pattern)?)))
    }

    pub fn split<'a>(&self, text: &'a str) -> Result<Vec<&'a str>, fancy_regex::Error> {
        let mut out = Vec::new();
        self.for_each(text, |piece| out.push(piece))?;
        Ok(out)
    }

    /// Calls `f` on every chunk, in order. Only the regex arm can fail (a runtime
    /// error such as fancy-regex's backtrack limit); it never silently truncates.
    pub fn for_each<'a>(
        &self,
        text: &'a str,
        mut f: impl FnMut(&'a str),
    ) -> Result<(), fancy_regex::Error> {
        match self {
            Splitter::Cl100k => cl100k::for_each(text, f),
            Splitter::O200k => o200k::for_each(text, f),
            Splitter::Whole => {
                if !text.is_empty() {
                    f(text)
                }
            }
            Splitter::Regex(re) => {
                for m in re.find_iter(text) {
                    let m = m?;
                    if !m.as_str().is_empty() {
                        f(m.as_str());
                    }
                }
            }
        }
        Ok(())
    }
}

/// Contraction suffixes after the apostrophe, in alternation order.
pub(crate) const CL100K_CONTRACTIONS: &[&str] = &["s", "d", "m", "t", "ll", "ve", "re"];
pub(crate) const O200K_CONTRACTIONS: &[&str] = &["s", "t", "re", "ve", "m", "ll", "d"];

/// `(?i:...)` over ASCII letters. Under Unicode simple case folding the only extra
/// member for these letters is U+017F LATIN SMALL LETTER LONG S, which folds to `s`.
#[inline]
fn folds_to(c: char, letter: u8) -> bool {
    c.to_ascii_lowercase() == letter as char || (letter == b's' && c == '\u{17f}')
}

/// Match one of `alternatives` (case-insensitively) starting at byte `j`; returns the end.
pub(crate) fn contraction(cur: &Cursor, j: usize, alternatives: &[&str]) -> Option<usize> {
    'alt: for alt in alternatives {
        let mut end = j;
        let mut chars = cur.text[j..].chars();
        for &letter in alt.as_bytes() {
            match chars.next() {
                Some(c) if folds_to(c, letter) => end += c.len_utf8(),
                _ => continue 'alt,
            }
        }
        return Some(end);
    }
    None
}

/// Char-level cursor helpers shared by the hand-written matchers.
pub(crate) struct Cursor<'a> {
    pub text: &'a str,
}

impl<'a> Cursor<'a> {
    #[inline]
    pub fn at(&self, i: usize) -> Option<char> {
        self.text[i..].chars().next()
    }

    /// Advance from `i` while `pred` holds; returns the end byte offset.
    #[inline]
    pub fn run(&self, mut i: usize, pred: impl Fn(char) -> bool) -> usize {
        for c in self.text[i..].chars() {
            if !pred(c) {
                break;
            }
            i += c.len_utf8();
        }
        i
    }
}

#[cfg(test)]
pub(crate) mod testutil {
    use super::*;

    pub fn regex_split(pattern: &str, text: &str) -> Vec<String> {
        Splitter::regex(pattern).unwrap().split(text).unwrap().into_iter().map(String::from).collect()
    }

    pub fn check(splitter: &Splitter, pattern: &str, text: &str) {
        let got: Vec<String> = splitter.split(text).unwrap().into_iter().map(String::from).collect();
        assert_eq!(got, regex_split(pattern, text), "input {text:?}");
    }
}
