# bytepair

A byte-level BPE tokenizer written from scratch in pure Python. It trains its own vocabularies, and it rebuilds OpenAI's `cl100k_base` (GPT-4) and `o200k_base` (GPT-4o) from their published ranks, reproducing tiktoken's output token for token.

```text
$ python scripts/parity.py
cl100k_base: 4,067,838 tokens over 11.4 MB, 0 mismatches
o200k_base:  3,391,387 tokens over 11.4 MB, 0 mismatches
```

### Highlights
- **Byte-identical to tiktoken:** 7.46M tokens across 11.4 MB of English, German, French, Spanish, Chinese, Japanese, Python source and a 1.5 MB emoji/whitespace fuzz file, with **0 mismatches** and lossless decode on both encodings.
- **Over 200x faster training than minbpe** (0.25-0.26 s vs 85-99 s, vocab 1024 on 752 KB, 3 runs), and 24-33x faster than the textbook recount-every-merge loop, with identical merges.
- **2.6-3.0x faster encoding** than the textbook per-chunk loop (4.6-5.0 MB/s cold, 12.4-13.1 MB/s warm on cl100k, single thread, pure Python, 3 runs).
- **Found a real divergence:** HF `tokenizers` ports of cl100k/o200k differ from tiktoken on untrusted text, because they turn a literal `<|endoftext|>` typed by a user into the real control token.

**Python · regex · tiktoken (reference) · HF tokenizers (reference) · pytest**

## Overview

Every LLM starts by turning text into integers, and many strange model behaviours (bad arithmetic, trouble with non-English text, prompt-injection via special tokens) trace back to that step. This project implements the whole pipeline: UTF-8 bytes, regex pre-splitting, greedy pair merging, special tokens and save/load. It then checks the implementation against the production tokenizers rather than against itself.

It follows Andrej Karpathy's [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) lecture and the exercise in [minbpe](https://github.com/karpathy/minbpe). The code is written independently. minbpe is used only as a training-speed baseline in the benchmark (cloned into `temp/`, not vendored).

## Results

All numbers are from `scripts/parity.py` and `scripts/benchmark.py`, measured 2026-09-26 on an AMD Ryzen 9 9950X3D, Python 3.12, tiktoken 0.14.0, tokenizers 0.23.2, one thread. Each cell is the min-max over 3 full benchmark runs (encode timings are each a median of 3). Raw output is in [`docs/parity.json`](docs/parity.json) and [`docs/benchmark.json`](docs/benchmark.json).

### Encode throughput (MB/s, 11.4 MB corpus)

| | cl100k_base | o200k_base |
|---|---:|---:|
| tiktoken (Rust) | 19.8-21.4 | 27.3-28.6 |
| **bytepair, warm chunk cache** | **12.4-13.1** | **10.2-11.5** |
| **bytepair, cold** | **4.6-5.0** | **4.1-4.3** |
| HF tokenizers, batch of lines, 1 thread | 3.1-3.3 | 3.2-3.3 |
| bytepair, textbook merge loop, cold | 1.7-1.8 | 1.5 |

Every bytepair run asserts its output equals tiktoken's before its time counts. HF is slower than its reputation here only because it is pinned to one thread; its speed comes from parallel batches.

### Training time

| | 752 KB, vocab 1024 | 11.4 MB, vocab 8192 |
|---|---:|---:|
| HF tokenizers (Rust) | 0.18-0.19 s | 4.2-4.4 s |
| **bytepair, incremental trainer** | **0.25-0.26 s** | **19.6-25.1 s** |
| bytepair, naive trainer | 8.0-8.5 s | n/a |
| minbpe `RegexTokenizer` | 85-99 s | n/a |

minbpe's time varies more between sessions than the others (an earlier session measured 61 s), but the ratio has stayed above 200x in every run.

Both trained tokenizers compress their training text to 2.968 bytes/token at vocab 1024. At 8192, bytepair reaches 3.337 and HF 3.352; the gap comes from tie-breaking and counting differences between the trainers.

### Compression (UTF-8 bytes per token, higher is better)

| | English | Code | German | Spanish | French | Japanese | Chinese | Fuzz | All |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cl100k_base | 4.35 | 4.36 | 3.25 | 3.32 | 3.34 | 2.62 | 2.03 | 1.87 | 2.80 |
| o200k_base | 4.38 | 4.33 | 3.68 | 3.72 | 3.78 | 3.08 | 2.93 | 2.29 | 3.35 |

(English = *Pride and Prejudice*, Japanese = *血笑記*.) o200k's larger vocabulary barely moves English or code, but it needs 30% fewer tokens for the same Chinese text and 17% fewer across the corpus.

## Engineering Highlights

- **Recovered GPT-4's merge tree from a flat rank table.** tiktoken ships only `{token bytes: rank}`. Re-running BPE on each token with merges capped below its own rank splits it into exactly its two parents, which rebuilds all 100,000 cl100k merges in 0.5 s and all 199,742 o200k merges in 2.2 s, including tiktoken's shuffled single-byte ranks.
- **Rank-array encoder:** the textbook loop rescans every pair and rebuilds the list on each merge. bytepair keeps adjacent-pair ranks in a list, merges the leftmost lowest-ranked pair and re-ranks only its two neighbours. It is 2.6-3.0x faster cold, and a test pins it to the textbook version.
- **Incremental trainer:** it deduplicates pre-split chunks by frequency, keeps running pair counts plus a pair-to-chunk index, and pops the best pair from a lazily invalidated heap. Each merge only re-counts chunks that contain the merged pair. Ties break on the smallest pair in both trainers, so the tests can require identical merge lists from the naive and incremental trainers.
- **Verification that can fail:** as a negative control, deleting the single merge that forms `" the"` makes the parity check fail (36,071 vs 34,893 tokens on the sample). The fuzz corpus mixes ZWJ emoji, flags, skin tones, combining accents, CRLF and lone CR, NBSP/ideographic/zero-width spaces, digit runs and uppercase contractions.
- **Special tokens handled like tiktoken:** `allowed_special` accepts `"all"`, `"none"`, `"none_raise"` (default, raises if a special string appears) or a set of strings. Delta-minimizing the HF divergence showed it comes entirely from added-token matching (`<|im_start|>`, `<|endoftext|>`, ...). With those strings removed, HF matches tiktoken exactly.

## Architecture

```text
text ──regex split──> chunks ──UTF-8──> byte ids ──BPE merges──> token ids
        (GPT-2/4 pattern)      (cache)    (tiktoken: byte-rank permutation)

src/bytepair/
  core.py        pair counting, pair replacement, tie-break, printable tokens
  trainers.py    naive and incremental (heap + index) merge learners
  base.py        Tokenizer: train, encode (rank-array), specials, decode, save/load
  tokenizers.py  ByteTokenizer, SplitTokenizer, TiktokenTokenizer (cl100k / o200k)
  cli.py         bytepair train | encode | decode
```

## Getting Started

```bash
python -m venv venv
venv/Scripts/python -m pip install -e ".[dev]"   # venv/bin/python on Linux/macOS

bytepair train data/sample.txt --vocab-size 512 --out temp/mytok
bytepair encode --model temp/mytok.model --text "It is a truth universally acknowledged"
bytepair encode --model cl100k_base --text "hello world"      # [15339, 1917]
bytepair decode --model o200k_base --ids "[3686, 220, 199999]" # hi <|endoftext|>
```

```python
from bytepair import SplitTokenizer, Cl100kTokenizer

tok = SplitTokenizer().train(open("data/sample.txt", encoding="utf-8").read(), vocab_size=1024)
tok.save("temp/pride")          # temp/pride.model + human-readable temp/pride.vocab
Cl100kTokenizer().encode("hello world")
```

Reproduce the numbers:

```bash
python scripts/fetch_corpus.py        # ~11.4 MB into temp/corpus (Project Gutenberg + stdlib + fuzz)
python scripts/parity.py              # docs/parity.json
git clone --depth 1 https://github.com/karpathy/minbpe temp/minbpe
python scripts/benchmark.py           # docs/benchmark.json
```

## Testing

`python -m pytest` runs 55 tests in about 10 s:
- the Wikipedia `aaabdaaabac` example
- round trips over ASCII, CJK, emoji and whitespace for all three splitting modes
- naive vs incremental merge equality on real text and 20 random corpora
- the fast chunk encoder against the textbook one
- save/load
- special-token modes
- cl100k and o200k parity with tiktoken on 16 edge cases plus the 150 KB sample
- the CLI

## What I Learned

- Tokenizer "correctness" is only meaningful against a reference. Two independent, well-used implementations (tiktoken and HF's ports) disagree on the same vocabulary, and the disagreement is a security-relevant one.
- In pure Python, the win comes from changing the algorithm, not from micro-tuning: deduplicating chunks and updating only what changed beat recounting by 24-33x.
- A merge's id is its rank, and any merge that uses it ranks later. That one invariant is what makes both the left-to-right encoder and the merge-tree recovery correct.

## License

MIT
