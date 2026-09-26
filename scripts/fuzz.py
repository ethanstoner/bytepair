"""Three-way fuzz: bytepair's Rust backend vs tiktoken vs the Python reference.

Random strings are drawn from a pool of the characters and sequences tokenizers trip on,
then every corpus line is checked too. Any mismatch is shrunk to a minimal input and
the script exits 1. Writes docs/fuzz.json.

    python scripts/fuzz.py [--n 1000000] [--seed 0] [--corpus DIR]
"""

import argparse
import json
import pathlib
import random
import sys
import time

import tiktoken

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from bytepair import TiktokenTokenizer  # noqa: E402
from fetch_corpus import EDGE_POOL  # noqa: E402

EXTRA_POOL = [
    "HELLOWorld", "helloWORLD", "ǅemo", "ǄEMO", "ʰ", "中", "́",
    "á", "'S", "'ſ", "'ſT", "'Ll", "'D", "'m", "/", "//", "a/", "!/\n",
    " \t\r\n", "  \r\n", "　x", "　　a", " ", "\u0085", " ",
    "12345", "٣٤٥٦", "½", "Ⅷ", "  ", " \n \n ",
]
POOL = EDGE_POOL + EXTRA_POOL
SPECIALS = ["<|endoftext|>", "<|fim_prefix|>", "<|endofprompt|>"]


def random_text(rng):
    return "".join(rng.choice(POOL) for _ in range(rng.randint(1, 30)))


def shrink(text, fails):
    """Delete characters while the failure persists."""
    changed = True
    while changed:
        changed = False
        for i in range(len(text)):
            candidate = text[:i] + text[i + 1:]
            if candidate and fails(candidate):
                text, changed = candidate, True
                break
    return text


def check_encoding(name, n, rng, corpus_lines):
    ref = tiktoken.get_encoding(name)
    rust = TiktokenTokenizer(name, backend="rust")
    py = TiktokenTokenizer(name, backend="python")

    def plain_fails(t):
        want = ref.encode_ordinary(t)
        return rust.encode_ordinary(t) != want or rust.decode(want) != ref.decode(want)

    def special_fails(t):
        return rust.encode(t, "all") != ref.encode(t, allowed_special="all")

    stats = {"random": 0, "random_python": 0, "with_specials": 0, "corpus_lines": 0}
    for k in range(n):
        text = random_text(rng)
        stats["random"] += 1
        if plain_fails(text):
            return stats, "plain", shrink(text, plain_fails)
        if k % 50 == 0:
            stats["random_python"] += 1
            if py.encode_ordinary(text) != ref.encode_ordinary(text):
                return stats, "python", shrink(text, lambda t: py.encode_ordinary(t) != ref.encode_ordinary(t))
        if k % 10 == 0:
            parts = [random_text(rng) for _ in range(3)]
            text = rng.choice(SPECIALS).join(parts[:2]) + rng.choice(SPECIALS) + parts[2]
            stats["with_specials"] += 1
            if special_fails(text):
                return stats, "specials", text
    for line in corpus_lines:
        stats["corpus_lines"] += 1
        if plain_fails(line):
            return stats, "corpus", shrink(line, plain_fails)
    return stats, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--corpus", default=str(ROOT / "temp" / "corpus"))
    args = ap.parse_args()

    corpus = pathlib.Path(args.corpus)
    files = sorted(corpus.glob("*.txt")) or [ROOT / "data" / "sample.txt"]
    lines = [ln for p in files for ln in p.read_bytes().decode("utf-8").splitlines(keepends=True)]

    report = {"n_random": args.n, "seed": args.seed, "corpus_files": [p.name for p in files],
              "measured_at": time.strftime("%Y-%m-%d"), "encodings": {}}
    failed = False
    for name in ("cl100k_base", "o200k_base"):
        t0 = time.perf_counter()
        stats, kind, example = check_encoding(name, args.n, random.Random(args.seed), lines)
        stats["mismatches"] = 0 if kind is None else 1
        stats["seconds"] = round(time.perf_counter() - t0, 1)
        report["encodings"][name] = stats
        print(name, stats, flush=True)
        if kind is not None:
            failed = True
            print(f"  MISMATCH ({kind}): {example!r}")
            report["encodings"][name]["example"] = {"kind": kind, "text": example}

    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "fuzz.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
