//! Byte-level BPE: pre-tokenizers, merging, encoding and training.

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod chars;
pub mod pretokenize;
