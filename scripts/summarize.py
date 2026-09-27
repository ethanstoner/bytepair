"""Print the README's results tables (min-max across runs) from docs/*.json.

Every number in the README's results section comes from this script's output, so nothing
is copied by hand.
"""

import json
import pathlib

DOCS = pathlib.Path(__file__).resolve().parents[1] / "docs"


def rng(values, fmt="{:.1f}"):
    lo, hi = min(values), max(values)
    a, b = fmt.format(lo), fmt.format(hi)
    return a if a == b else f"{a}-{b}"


def ratio_range(runs, get_a, get_b, fmt="{:.1f}x"):
    return rng([get_a(r) / get_b(r) for r in runs], fmt)


def main():
    bench = json.loads((DOCS / "benchmark.json").read_text(encoding="utf-8"))
    runs = bench["runs"]
    single = [r["single"] for r in runs]
    multi = [r["multi"] for r in runs]
    enc = ("cl100k_base", "o200k_base")

    print(f"Measured {bench['measured_at']} on {bench['cpu']} ({bench['logical_cores']} logical cores), "
          f"Python {bench['python']}, tiktoken {bench['tiktoken']}, tokenizers {bench['tokenizers']}; "
          f"{len(runs)} runs, corpus {bench['corpus_bytes'] / 1e6:.1f} MB.\n")

    print("### Encode, one thread (MB/s)\n")
    print("| | cl100k_base | o200k_base |\n|---|---:|---:|")
    rows = [
        ("**bytepair (Rust), warm cache**", "bytepair_rust_warm"),
        ("**bytepair (Rust)**", "bytepair_rust_cold"),
        ("tiktoken", "tiktoken"),
        ("bytepair (Python reference), warm cache", "bytepair_python_warm"),
        ("bytepair (Python reference)", "bytepair_python_cold"),
        ("HF tokenizers, batch of lines", "hf_tokenizers_batch_of_lines"),
        ("bytepair (Python), textbook merge loop", "bytepair_python_textbook_cold"),
    ]
    for label, key in rows:
        cells = [rng([s["encode_mb_per_s"][e][key] for s in single]) for e in enc]
        print(f"| {label} | {' | '.join(cells)} |")
    for e in enc:
        print(f"\n{e}: Rust cold vs tiktoken "
              + ratio_range(single, lambda s: s["encode_mb_per_s"][e]["bytepair_rust_cold"],
                            lambda s: s["encode_mb_per_s"][e]["tiktoken"], "{:.2f}x"))

    threads = multi[0]["threads"]
    docs = multi[0]["encode_mb_per_s"]["documents"]
    print(f"\n### Encode, {threads} threads, {docs} documents of ~64 KB (MB/s)\n")
    print("| | cl100k_base | o200k_base |\n|---|---:|---:|")
    for label, key in [("**bytepair (Rust) `encode_batch`**", "bytepair_rust_batch"),
                       ("tiktoken `encode_ordinary_batch`", "tiktoken_batch"),
                       ("HF tokenizers `encode_batch`", "hf_tokenizers_batch")]:
        cells = [rng([m["encode_mb_per_s"][e][key] for m in multi]) for e in enc]
        print(f"| {label} | {' | '.join(cells)} |")
    for e in enc:
        print(f"\n{e}: Rust batch vs tiktoken batch "
              + ratio_range(multi, lambda m: m["encode_mb_per_s"][e]["bytepair_rust_batch"],
                            lambda m: m["encode_mb_per_s"][e]["tiktoken_batch"]))

    print("\n### Training (seconds)\n")
    pride = [s["train"][0] for s in single]
    corpus = [s["train"][1] for s in single]
    par = [m["train"] for m in multi]
    print(f"| | {pride[0]['train_bytes'] / 1e3:.0f} KB, vocab {pride[0]['vocab_size']} "
          f"| {corpus[0]['train_bytes'] / 1e6:.1f} MB, vocab {corpus[0]['vocab_size']} |\n|---|---:|---:|")
    print(f"| **bytepair (Rust), 1 thread** | {rng([p['bytepair_rust_s'] for p in pride], '{:.2f}')} "
          f"| {rng([c['bytepair_rust_s'] for c in corpus], '{:.2f}')} |")
    print(f"| HF tokenizers, 1 thread | {rng([p['hf_tokenizers_s'] for p in pride], '{:.2f}')} "
          f"| {rng([c['hf_tokenizers_s'] for c in corpus], '{:.2f}')} |")
    print(f"| **bytepair (Rust), {threads} threads** | | {rng([p['bytepair_rust_parallel_s'] for p in par], '{:.2f}')} |")
    print(f"| HF tokenizers, {threads} threads | | {rng([p['hf_tokenizers_parallel_s'] for p in par], '{:.2f}')} |")
    print(f"| bytepair (Python reference) | {rng([p['bytepair_python_s'] for p in pride], '{:.2f}')} "
          f"| {rng([c['bytepair_python_s'] for c in corpus])} |")
    print(f"| bytepair (Python), naive recount | {rng([p['bytepair_python_naive_s'] for p in pride])} | |")
    print(f"| minbpe `RegexTokenizer` | {rng([p['minbpe_s'] for p in pride])} | |")
    print("\nRust vs HF, 1 thread, corpus: "
          + ratio_range(corpus, lambda c: c["hf_tokenizers_s"], lambda c: c["bytepair_rust_s"], "{:.2f}x"))
    print("Rust vs minbpe, pride: "
          + ratio_range(pride, lambda p: p["minbpe_s"], lambda p: p["bytepair_rust_s"], "{:.0f}x"))
    print("Rust vs Python reference, corpus: "
          + ratio_range(corpus, lambda c: c["bytepair_python_s"], lambda c: c["bytepair_rust_s"], "{:.1f}x"))
    print(f"bytes/token after training: bytepair {corpus[0]['bytepair_bytes_per_token']}, "
          f"HF {corpus[0]['hf_bytes_per_token']}")

    split = DOCS / "split_bench.json"
    if split.exists():
        s = json.loads(split.read_text(encoding="utf-8"))
        for e in ("cl100k", "o200k"):
            print(f"\nsplitter {e}: hand-written {s[e]['hand_written_mb_s']} MB/s vs fancy-regex "
                  f"{s[e]['fancy_regex_mb_s']} MB/s ({s[e]['speedup']}x)")

    fuzz = json.loads((DOCS / "fuzz.json").read_text(encoding="utf-8"))
    for e, st in fuzz["encodings"].items():
        print(f"\nfuzz {e}: {st['random']:,} random + {st['with_specials']:,} with specials + "
              f"{st['corpus_lines']:,} corpus lines ({st['random_python']:,} also via Python): "
              f"{st['mismatches']} mismatches")
    parity = json.loads((DOCS / "parity.json").read_text(encoding="utf-8"))
    for e, st in parity["encodings"].items():
        print(f"parity {e}: {st['tokens']:,} tokens, mismatches {st['mismatches']}")


if __name__ == "__main__":
    main()
