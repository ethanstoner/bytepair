//! Unicode character classes used by the hand-written pre-tokenizers.
//!
//! The tables come from `regex-syntax`, the same parser fancy-regex (and so tiktoken)
//! uses, so `\p{L}` here means exactly what it means in the reference regexes.

use std::sync::OnceLock;

use regex_syntax::hir::{Class, HirKind};

pub const LETTER: u16 = 1 << 0;
pub const UPPER: u16 = 1 << 1; // Lu
pub const LOWER: u16 = 1 << 2; // Ll
pub const TITLE: u16 = 1 << 3; // Lt
pub const MODIFIER: u16 = 1 << 4; // Lm
pub const OTHER_LETTER: u16 = 1 << 5; // Lo
pub const MARK: u16 = 1 << 6; // M
pub const NUMBER: u16 = 1 << 7; // N
pub const SPACE: u16 = 1 << 8; // \s

/// o200k's `[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]`.
pub const O200K_UPPER: u16 = UPPER | TITLE | MODIFIER | OTHER_LETTER | MARK;
/// o200k's `[\p{Ll}\p{Lm}\p{Lo}\p{M}]`.
pub const O200K_LOWER: u16 = LOWER | MODIFIER | OTHER_LETTER | MARK;

const CLASSES: [(&str, u16); 9] = [
    (r"\p{L}", LETTER),
    (r"\p{Lu}", UPPER),
    (r"\p{Ll}", LOWER),
    (r"\p{Lt}", TITLE),
    (r"\p{Lm}", MODIFIER),
    (r"\p{Lo}", OTHER_LETTER),
    (r"\p{M}", MARK),
    (r"\p{N}", NUMBER),
    (r"\s", SPACE),
];

fn table() -> &'static [u16] {
    static TABLE: OnceLock<Vec<u16>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut flags = vec![0u16; 0x11_0000];
        for (pattern, bit) in CLASSES {
            let hir = regex_syntax::parse(pattern).expect("class pattern parses");
            let HirKind::Class(Class::Unicode(class)) = hir.kind() else {
                panic!("{pattern} is not a unicode class");
            };
            for range in class.ranges() {
                for cp in range.start() as u32..=range.end() as u32 {
                    flags[cp as usize] |= bit;
                }
            }
        }
        flags
    })
}

#[inline]
pub fn flags(c: char) -> u16 {
    table()[c as usize]
}

#[inline]
pub fn is(c: char, mask: u16) -> bool {
    flags(c) & mask != 0
}

#[inline]
pub fn is_letter(c: char) -> bool {
    is(c, LETTER)
}

#[inline]
pub fn is_number(c: char) -> bool {
    is(c, NUMBER)
}

#[inline]
pub fn is_space(c: char) -> bool {
    is(c, SPACE)
}

#[inline]
pub fn is_newline(c: char) -> bool {
    c == '\r' || c == '\n'
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classes() {
        assert!(is_letter('a') && is('a', LOWER) && !is('a', UPPER));
        assert!(is('Z', UPPER));
        assert!(is_number('1') && is_number('٣'));
        assert!(is('\u{301}', MARK) && !is_letter('\u{301}'));
        assert!(is('ǅ', TITLE));
        assert!(is('ʰ', MODIFIER) && is('中', OTHER_LETTER));
        assert!(is_space('\u{3000}') && is_space('\u{a0}') && is_space('\u{85}'));
        assert!(!is_space('\u{200b}') && !is_space('\u{feff}'));
        assert!(!is_letter('!') && !is_number('!') && !is_space('!'));
    }
}
