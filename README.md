# bytepair

A byte-level BPE tokenizer with a Rust engine and a readable pure-Python reference. Both produce identical output. It rebuilds OpenAI's `cl100k_base` (GPT-4) and `o200k_base` (GPT-4o) from their published ranks, matches tiktoken token for token, and encodes faster than tiktoken on one thread.

```text
                      single-thread encode, 11.4 MB corpus (MB/s, 3 runs)
bytepair (Rust)       ███████████████████████████████████  32.3-45.2
tiktoken              ████████████████████                  20.5
HF tokenizers         ███                                    2.1-3.2
                      identical output to tiktoken: 4,067,838 tokens, 0 mismatches
```

### Highlights
- **1.6-2.2x faster than tiktoken** encoding cl100k_base on one thread, and 1.55-1.8x on o200k_base. The output is byte-identical: 7.46M tokens over 11.4 MB, 0 mismatches.
- **3.2-4.9x faster BPE training than HF `tokenizers`**, single-threaded: 1.43-1.60 s vs 5.0-7.0 s on 11.4 MB at vocab 8192. It learns exactly the same merges as the pure-Python reference.
- **Hand-written pre-tokenizers 15.5x (cl100k) and 20x (o200k) faster than the regex engine** they replace, verified equal on the whole corpus, 1M fuzz strings and property tests.
- **Found a real divergence:** HF `tokenizers` ports of cl100k/o200k turn a literal `<|endoftext|>` in untrusted user text into the real control token. tiktoken refuses it by default, and so does bytepair.

**Rust · PyO3 · rayon · Python · regex · tiktoken and HF tokenizers (as references) · pytest · proptest**

## Overview

Every LLM starts by turning text into integers. Many odd model behaviours trace back to that step: bad arithmetic, uneven cost across languages, prompt injection through special tokens.

This project implements the whole pipeline twice:
- UTF-8 bytes
- regex pre-splitting
- greedy pair merging
- special tokens
- training
- save/load

The Python version is the readable reference; the Rust version is the fast one. Every test runs against both, and both are checked against the production tokenizers rather than against each other alone.

It started from Andrej Karpathy's [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) lecture and the exercise in [minbpe](https://github.com/karpathy/minbpe). The code is written independently; minbpe appears only as a training-time baseline.

## Results

All numbers come from `scripts/summarize.py` over [`docs/benchmark.json`](docs/benchmark.json), [`docs/split_bench.json`](docs/split_bench.json), [`docs/parity.json`](docs/parity.json) and [`docs/fuzz.json`](docs/fuzz.json).

Measurement setup:
- **Machine:** 2026-09-26, AMD Ryzen 9 9950X3D (32 logical cores), Python 3.12.10, tiktoken 0.14.0, tokenizers 0.23.2. CPU load was under 15% for a minute before starting and 4% at the end.
- **Runs:** each cell is the min-max over 3 full runs. Encode timings are medians of 3 within each run.
- **Checks:** every bytepair encode is asserted equal to tiktoken's output before its time counts.
- **Corpus:** 11.4 MB of English, German, French, Spanish, Chinese and Japanese books, Python stdlib source, and an emoji/whitespace fuzz file.

### Encode, one thread (MB/s)

| | cl100k_base | o200k_base |
|---|---:|---:|
| **bytepair (Rust), warm chunk cache** | **53.9-69.4** | **51.7-67.5** |
| **bytepair (Rust), cold** | **32.3-45.2** | **30.1-40.4** |
| tiktoken | 20.5 | 18.4-25.6 |
| bytepair (Python reference), warm cache | 8.1-12.9 | 6.9-10.6 |
| bytepair (Python reference), cold | 3.1-4.7 | 2.6-4.1 |
| HF tokenizers, batch of lines | 2.1-3.2 | 2.1-3.1 |
| bytepair (Python), textbook merge loop | 1.1-1.8 | 0.9-1.5 |

The per-run Rust-vs-tiktoken ratio is 1.58-2.20x on cl100k and 1.55-1.78x on o200k.

### Encode, 32 threads (MB/s)

The batch is 137 documents of about 64 KB each, the same list for all three libraries.

| | cl100k_base | o200k_base |
|---|---:|---:|
| **bytepair (Rust) `encode_batch`** | **84.7-123.2** | **120.0-156.5** |
| tiktoken `encode_ordinary_batch(num_threads=32)` | 99.1-103.6 | 100.7-119.8 |
| HF tokenizers `encode_batch` | 23.2-25.6 | 20.7-23.3 |

With many threads the advantage shrinks. bytepair's per-run ratio to tiktoken is 0.8-1.2x on cl100k (effectively even) and 1.2-1.3x on o200k. Each worker starts with a cold chunk cache, and the Rust batch timings vary more between runs than tiktoken's.

### Training (seconds)

| | 752 KB, vocab 1024 | 11.4 MB, vocab 8192 |
|---|---:|---:|
| **bytepair (Rust), 1 thread** | **0.017-0.023** | **1.43-1.60** |
| HF tokenizers, 1 thread | 0.20-0.34 | 5.02-6.98 |
| **bytepair (Rust), 32 threads** | | **1.58-1.78** |
| HF tokenizers, 32 threads | | 5.75-7.46 |
| bytepair (Python reference, incremental) | 0.25-0.39 | 20.2-27.1 |
| bytepair (Python, naive recount) | 15.6-19.2 | |
| minbpe `RegexTokenizer` | 121-135 | |

Every Rust run asserts the same merges as the Python reference. Compression after training is 3.337 bytes/token for bytepair and 3.352 for HF; the gap comes from tie-breaking.

Three caveats:
- Parallel pair counting does not help at this size, so the 32-thread Rust row is no faster than 1 thread.
- Pure-Python timings (minbpe, the Python reference) varied up to 2x between sessions on this machine, while the Rust, tiktoken and HF timings did not. Ratios against Python baselines are therefore only compared within a run.
- Within these runs, Rust training is over 5,000x faster than minbpe and 13-19x faster than the Python reference.

### Correctness

| Check | Scope | Result |
|---|---|---|
| Parity vs tiktoken | 11.4 MB corpus, both encodings, Rust and Python backends | 0 mismatches, lossless decode |
| Three-way fuzz (Rust vs tiktoken vs Python) | per encoding: 1M random edge-case strings, 100k with special tokens spliced in, all 210,967 corpus lines; 20k also through Python | 0 mismatches |
| Splitters vs `regex.findall` | every corpus file and line, both patterns | 0 mismatches |
| Splitters vs fancy-regex | 20,000 property-test strings per pattern; whole corpus in `split_bench` | 0 mismatches |
| Rust trainer vs Python trainer | 11.4 MB at vocab 8192 (7,936 merges), random Unicode text, 4 split patterns | identical merges |

As a negative control, an injected one-character bug is caught by the fuzz within 4 strings and shrunk to that character.

### Compression (UTF-8 bytes per token, higher is better)

| | English | Code | German | Spanish | French | Japanese | Chinese | Fuzz | All |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cl100k_base | 4.35 | 4.36 | 3.25 | 3.32 | 3.34 | 2.62 | 2.03 | 1.87 | 2.80 |
| o200k_base | 4.38 | 4.33 | 3.68 | 3.72 | 3.78 | 3.08 | 2.93 | 2.29 | 3.35 |

o200k's larger vocabulary barely changes English or code. It needs 30% fewer tokens for the same Chinese text, and 17% fewer across the whole corpus.

## Engineering Highlights

- **Replaced the regex engine with hand-written matchers (15.5-20x faster splitting).** tiktoken splits text with fancy-regex. bytepair implements each alternative of the cl100k and o200k patterns directly:
  - possessive vs backtracking quantifiers reproduced exactly;
  - `\s+(?!\S)` backing off one *character*, not one byte;
  - `(?i)` contractions including the long-s fold (`'ſ` matches `'s`).

  o200k's first alternative backtracks for real, because modifier letters, other letters and combining marks belong to both its "upper" and "lower" classes. Unicode classes are built from `regex-syntax`, the same tables tiktoken's regex engine uses. Other patterns fall back to fancy-regex.
- **Verified whole-token shortcut.** When the encoder is built, every vocabulary token is run through BPE. Tokens that come back as exactly themselves (all 100,256 cl100k and 199,998 o200k tokens) go in a lookup table, and chunks matching one skip merging. For custom vocabularies, tokens that fail the check are left out, so the shortcut never changes output.
- **Recovered GPT-4's merge tree from a flat rank table.** tiktoken ships only `{token bytes: rank}`. Re-running BPE on each token, with merges capped below its own rank, splits it into exactly its two parents. Rust rebuilds all 199,742 o200k merges in parallel.
- **Incremental trainer, identical in both languages.** It uses:
  - deduplicated chunks with frequencies
  - a pair-to-chunk index
  - a lazily invalidated max-heap
  - smallest-pair tie-breaking

  Per merge, only chunks containing that pair are revisited. Profiling showed the early merges touch hundreds of thousands of chunks, so Rust diffs sorted pair buffers instead of building hash maps per chunk. That cut training time by about 30% in a before/after profiling run.
- **Rank-array merging.** Adjacent-pair ranks are kept in an array. Each step merges the leftmost lowest-ranked pair and re-ranks only its neighbours. In Python this is 1.9-2.8x faster than the textbook recount loop. It relies on one invariant: a merge's id is its rank, and anything built on it ranks later.
- **Parallel batch encoding.** It releases the GIL and gives each rayon worker a private chunk cache. Benchmark batches are ~64 KB documents: tiktoken's batch API schedules one thread-pool task per item, so single lines would measure task overhead rather than encoding.
- **Special tokens handled like tiktoken:**
  - `allowed_special` takes `"all"`, `"none"`, `"none_raise"` (the default) or a set of names;
  - the text is split leftmost-first in registration order;
  - `$` is applied per segment;
  - every error path raises `ValueError` in both backends.
- **Benchmarks that can't lie by accident:**
  - separate single- and multi-thread subprocesses, because thread pools size themselves once per process;
  - Python rows pinned to `backend="python"`, so `auto` can't quietly time Rust;
  - an output check before every timed encode;
  - ranges over repeated runs;
  - tables generated from JSON.

  An earlier single run once overstated a speedup; that's why these rules exist.

## Architecture

```text
text ─ specials split ─ pre-tokenizer ─ chunk ─┬─ single byte / whole token ─ id
       (leftmost-first)  (hand-written or      ├─ cache hit ─ ids
                          fancy-regex)         └─ rank-array BPE ─ ids

crates/bytepair-core/   Rust: chars (Unicode classes), pretokenize/{cl100k,o200k}, bpe,
                        encoder (Encoder, specials, cache, batch), train
crates/bytepair-py/     PyO3 module bytepair._core
src/bytepair/           Python package: backend switch + the pure-Python reference
scripts/                fetch_corpus, parity, fuzz, benchmark, summarize, build.sh
```

## Getting Started

You need Rust (stable) and Python 3.10+.

```bash
python -m venv venv && source venv/bin/activate      # venv\Scripts\activate on Windows
pip install maturin pytest regex tiktoken tokenizers
maturin develop --release                            # Windows without MSVC: scripts/build.sh

bytepair train data/sample.txt --vocab-size 512 --out temp/mytok
bytepair encode --model cl100k_base --text "hello world"          # [15339, 1917]
bytepair --backend python encode --model o200k_base --text "hi"   # pure-Python reference
```

```python
from bytepair import SplitTokenizer, Cl100kTokenizer

tok = SplitTokenizer().train(open("data/sample.txt", encoding="utf-8").read(), vocab_size=1024)
tok.save("temp/pride")                      # .model plus a human-readable .vocab
Cl100kTokenizer().encode("hello world")     # backend="auto": Rust if built, else Python
```

To reproduce the results (on an idle machine):

```bash
python scripts/fetch_corpus.py     # ~11.4 MB into temp/corpus (Project Gutenberg + stdlib + fuzz)
python scripts/parity.py           # docs/parity.json
python scripts/fuzz.py             # docs/fuzz.json
git clone --depth 1 https://github.com/karpathy/minbpe temp/minbpe
python scripts/benchmark.py 3      # docs/benchmark.json
cargo run --release -p bytepair-core --example split_bench -- temp/corpus > docs/split_bench.json
python scripts/summarize.py        # the tables above, from the JSON
```

## Testing

`pytest` runs 136 tests, most of them against both backends:
- tiktoken parity on edge cases and the sample
- splitters vs `regex.findall`
- Rust vs Python trainer merges
- special-token modes and tie-breaking
- `ValueError` on every error path
- lossy decode matching Python's `errors="replace"`
- the `auto` fallback
- the CLI

`cargo test -p bytepair-core` runs 13 Rust tests, including property tests of both splitters against fancy-regex and of the trainer against a naive recount. CI (GitHub Actions, Linux) runs clippy with `-D warnings`, both test suites and a 100k-string fuzz.

## What I Learned

- A tokenizer is only "correct" relative to a reference. Two widely used implementations of the same vocabulary disagree, and the disagreement matters for security.
- The biggest speedup came from removing work, not from making it faster: skipping the regex engine, and skipping BPE for chunks that are already tokens. Both were only safe with an oracle to test against.
- Benchmarks need an idle machine, repeated runs and ranges. One run under another project's load halved every baseline, and interpreted-Python timings drift between sessions far more than native code does.

## License

MIT
