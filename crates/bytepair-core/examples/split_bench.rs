//! Pre-tokenizer throughput: hand-written splitters vs fancy-regex on the same pattern.
//!
//!     cargo run --release --example split_bench -- temp/corpus
//!
//! Prints one JSON object; asserts both splitters produce identical chunks first.

use std::time::Instant;
use std::{env, fs};

use bytepair_core::pretokenize::{Splitter, CL100K_PATTERN, O200K_PATTERN};

fn median_secs(mut f: impl FnMut() -> usize, repeats: usize) -> f64 {
    let mut times: Vec<f64> = (0..repeats)
        .map(|_| {
            let t = Instant::now();
            std::hint::black_box(f());
            t.elapsed().as_secs_f64()
        })
        .collect();
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[repeats / 2]
}

fn main() {
    let dir = env::args().nth(1).unwrap_or_else(|| "temp/corpus".into());
    let mut paths: Vec<_> = fs::read_dir(&dir)
        .expect("corpus dir")
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "txt"))
        .collect();
    paths.sort();
    let text: String = paths.iter().map(|p| fs::read_to_string(p).unwrap()).collect();
    let mb = text.len() as f64 / 1e6;

    let mut rows = Vec::new();
    for (name, pattern, fast) in
        [("cl100k", CL100K_PATTERN, Splitter::Cl100k), ("o200k", O200K_PATTERN, Splitter::O200k)]
    {
        let regex = Splitter::regex(pattern).unwrap();
        assert_eq!(fast.split(&text).unwrap(), regex.split(&text).unwrap(), "{name}: splitters disagree");
        let count = |s: &Splitter| {
            let mut n = 0;
            s.for_each(&text, |_| n += 1).unwrap();
            n
        };
        let hand = mb / median_secs(|| count(&fast), 5);
        let re = mb / median_secs(|| count(&regex), 5);
        rows.push(format!(
            "\"{name}\": {{\"hand_written_mb_s\": {hand:.1}, \"fancy_regex_mb_s\": {re:.1}, \"speedup\": {:.2}}}",
            hand / re
        ));
    }
    println!("{{\"corpus_mb\": {mb:.2}, {}}}", rows.join(", "));
}
