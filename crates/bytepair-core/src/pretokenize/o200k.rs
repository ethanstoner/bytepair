//! Hand-written equivalent of the o200k_base split pattern:
//!
//! ```text
//! [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?
//! |[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?
//! |\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+
//! ```
//!
//! Unlike cl100k nothing here is possessive, and Lm/Lo/M sit in both the "upper" and
//! "lower" classes, so the first alternative has to reproduce the regex's backtracking.

use super::{contraction, Cursor, O200K_CONTRACTIONS};
use crate::chars::{is, is_letter, is_newline, is_number, is_space, O200K_LOWER, O200K_UPPER};

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
fn upper(c: char) -> bool {
    is(c, O200K_UPPER)
}

#[inline]
fn lower(c: char) -> bool {
    is(c, O200K_LOWER)
}

#[inline]
fn is_prefix(c: char) -> bool {
    !is_newline(c) && !is_letter(c) && !is_number(c)
}

#[inline]
fn is_punct(c: char) -> bool {
    !is_space(c) && !is_letter(c) && !is_number(c)
}

/// Optional `(?i:'s|'t|'re|'ve|'m|'ll|'d)` at `j`.
fn with_suffix(cur: &Cursor, j: usize) -> usize {
    if cur.at(j) == Some('\'') {
        if let Some(end) = contraction(cur, j + 1, O200K_CONTRACTIONS) {
            return end;
        }
    }
    j
}

/// `UPPER* LOWER+` from `s`, backtracking the greedy UPPER* as the regex does.
fn upper_star_lower_plus(cur: &Cursor, s: usize) -> Option<usize> {
    let k = cur.run(s, upper);
    // Greedy UPPER* first; LOWER+ must then start at the chosen split point.
    if cur.at(k).is_some_and(lower) {
        return Some(cur.run(k, lower));
    }
    for (off, c) in cur.text[s..k].char_indices().rev() {
        if lower(c) {
            return Some(cur.run(s + off, lower));
        }
    }
    None
}

/// `UPPER+ LOWER*` from `s`.
fn upper_plus_lower_star(cur: &Cursor, s: usize) -> Option<usize> {
    let k = cur.run(s, upper);
    (k > s).then(|| cur.run(k, lower))
}

fn letters(cur: &Cursor, i: usize, c: char, body: fn(&Cursor, usize) -> Option<usize>) -> Option<usize> {
    // The optional prefix is greedy: try with it, then without.
    if is_prefix(c) {
        if let Some(end) = body(cur, i + c.len_utf8()) {
            return Some(with_suffix(cur, end));
        }
    }
    body(cur, i).map(|end| with_suffix(cur, end))
}

fn match_at(cur: &Cursor, i: usize, c: char) -> Option<usize> {
    if let Some(end) = letters(cur, i, c, upper_star_lower_plus) {
        return Some(end);
    }
    if let Some(end) = letters(cur, i, c, upper_plus_lower_star) {
        return Some(end);
    }

    // \p{N}{1,3}
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

    // ' ?[^\s\p{L}\p{N}]+[\r\n/]*'
    let p = if c == ' ' { i + 1 } else { i };
    let q = cur.run(p, is_punct);
    if q > p {
        return Some(cur.run(q, |d| is_newline(d) || d == '/'));
    }

    if is_space(c) {
        let j = cur.run(i, is_space);
        let run = &cur.text[i..j];
        // \s*[\r\n]+: backtracks to the last newline in the run.
        if let Some(k) = run.rfind(['\r', '\n']) {
            return Some(i + k + 1);
        }
        // \s+(?!\S): at end of text the whole run; before a non-space, give back one char.
        if j == cur.text.len() {
            return Some(j);
        }
        let last = run.chars().next_back().map_or(0, char::len_utf8);
        if j - last > i {
            return Some(j - last);
        }
        // \s+
        return Some(j);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::super::testutil::check;
    use super::super::{Splitter, O200K_PATTERN};
    use proptest::prelude::*;

    fn eq(text: &str) {
        check(&Splitter::O200k, O200K_PATTERN, text);
    }

    #[test]
    fn known_cases() {
        for s in [
            "HELLOWorld",
            "helloWORLD",
            "ǅemo ǄEMO",
            "e\u{301}clair \u{301}x",
            "DON'T it'S we'ſ",
            "path/to//file",
            "a//\n",
            "x \n\n y",
            "٣٤٥٦",
            "  \n \n  x",
            "trailing  ",
            "ʰa中文ʰA",
            "",
        ] {
            eq(s);
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(20_000))]
        #[test]
        fn matches_regex(s in "[aAZéǄǅʰ中1' \t\r\n!/\u{3000}\u{301}😀ſ]{0,24}") {
            eq(&s);
        }
    }
}
