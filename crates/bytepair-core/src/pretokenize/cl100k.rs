//! Hand-written equivalent of the cl100k_base split pattern:
//!
//! ```text
//! '(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s
//! ```
//!
//! Alternatives are tried in order at each position and the first that matches wins,
//! as in the regex engine.

use super::Cursor;
use crate::chars::{is_letter, is_newline, is_number, is_space};

pub fn for_each<'a>(text: &'a str, mut f: impl FnMut(&'a str)) {
    let cur = Cursor { text };
    let mut i = 0;
    while let Some(c) = cur.at(i) {
        let end = match_at(&cur, i, c).unwrap_or(i + c.len_utf8());
        f(&text[i..end]);
        i = end;
    }
}

#[inline]
fn is_punct(c: char) -> bool {
    !is_space(c) && !is_letter(c) && !is_number(c)
}

/// `[^\r\n\p{L}\p{N}]`: the optional prefix before a letter run.
#[inline]
fn is_prefix(c: char) -> bool {
    !is_newline(c) && !is_letter(c) && !is_number(c)
}

fn match_at(cur: &Cursor, i: usize, c: char) -> Option<usize> {
    let len = cur.text.len();

    // '(?i:[sdmt]|ll|ve|re)
    if c == '\'' {
        if let Some(end) = super::contraction(cur, i + 1, super::CL100K_CONTRACTIONS) {
            return Some(end);
        }
    }

    // [^\r\n\p{L}\p{N}]?+\p{L}++  (possessive prefix: no retry without it, and a
    // retry could only succeed if the prefix char were itself a letter)
    if is_letter(c) {
        return Some(cur.run(i, is_letter));
    }
    if is_prefix(c) {
        let j = i + c.len_utf8();
        if cur.at(j).is_some_and(is_letter) {
            return Some(cur.run(j, is_letter));
        }
    }

    // \p{N}{1,3}+
    if is_number(c) {
        let mut end = i;
        for d in cur.text[i..].chars().take(3) {
            if !is_number(d) {
                break;
            }
            end += d.len_utf8();
        }
        return Some(end);
    }

    // ' ?[^\s\p{L}\p{N}]++[\r\n]*+'. Without the optional space, a space start could
    // never match, so only one attempt is needed.
    let p = if c == ' ' { i + 1 } else { i };
    let q = cur.run(p, is_punct);
    if q > p {
        return Some(cur.run(q, is_newline));
    }

    if is_space(c) {
        let j = cur.run(i, is_space);
        // \s++$
        if j == len {
            return Some(len);
        }
        // \s*[\r\n]: backtracks to the last newline in the run.
        let run = &cur.text[i..j];
        if let Some(k) = run.rfind(['\r', '\n']) {
            return Some(i + k + 1);
        }
        // \s+(?!\S): the run is followed by a non-space, so give back its last char.
        let last = run.chars().next_back().map_or(0, char::len_utf8);
        if j - last > i {
            return Some(j - last);
        }
        // \s
        return Some(i + c.len_utf8());
    }
    None
}

#[cfg(test)]
mod tests {
    use super::super::testutil::check;
    use super::super::{Splitter, CL100K_PATTERN};
    use proptest::prelude::*;

    fn eq(text: &str) {
        check(&Splitter::Cl100k, CL100K_PATTERN, text);
    }

    #[test]
    fn known_cases() {
        let got = Splitter::Cl100k.split("I'M don't").unwrap();
        assert_eq!(got, ["I", "'M", " don", "'t"]);
        assert_eq!(Splitter::Cl100k.split("1234567").unwrap(), ["123", "456", "7"]);
        for s in [
            "hello   world",
            "a\r\n\r\n  b",
            "x  \n",
            "!!!\n\nabc",
            "   ",
            "\u{3000}\u{3000}x",
            "it'ſ THE'LL we'VE",
            "\t\thello",
            " !! ?",
            "e\u{301}clair 1\u{301}",
            "\u{a0}\u{a0}\n x",
            "",
        ] {
            eq(s);
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(20_000))]
        #[test]
        fn matches_regex(s in "[aZé1٣' \t\r\n!/\u{3000}\u{301}😀ſ]{0,24}") {
            eq(&s);
        }
    }
}
