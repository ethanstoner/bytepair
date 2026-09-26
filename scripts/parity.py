"""Encode the whole corpus with our rebuilt encodings and with tiktoken; count mismatches.

Also reports compression (UTF-8 bytes per token) per file. Writes docs/parity.json.
"""

import json
import pathlib
import sys
import time

import tiktoken

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bytepair import TiktokenTokenizer  # noqa: E402

CORPUS = ROOT / "temp" / "corpus"


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def main(names=("cl100k_base", "o200k_base")):
    files = sorted(CORPUS.glob("*.txt"))
    if not files:
        sys.exit("no corpus; run scripts/fetch_corpus.py first")
    texts = {p.stem: p.read_bytes().decode("utf-8") for p in files}
    total_bytes = sum(len(t.encode("utf-8")) for t in texts.values())
    report = {"corpus_bytes": total_bytes, "encodings": {}}

    for name in names:
        ours = TiktokenTokenizer(name)
        ref = tiktoken.get_encoding(name)
        rows = {}
        tokens = mismatches = 0
        for stem, text in texts.items():
            a = ours.encode(text, allowed_special="none")
            b = ref.encode(text, disallowed_special=())
            same = a == b
            round_trip = ours.decode(a) == text
            if not same:
                i = first_diff(a, b)
                print(f"  MISMATCH {name} {stem} at token {i}: {a[i:i+5]} vs {b[i:i+5]}")
            mismatches += (not same) + (not round_trip)
            tokens += len(b)
            nbytes = len(text.encode("utf-8"))
            rows[stem] = {"bytes": nbytes, "tokens": len(b), "bytes_per_token": round(nbytes / len(b), 3),
                          "identical": same, "round_trip": round_trip}
            print(f"{name:12} {stem:28} {len(b):>9,} tokens  {nbytes / len(b):5.2f} B/tok  "
                  f"{'ok' if same and round_trip else 'FAIL'}", flush=True)
        report["encodings"][name] = {"tokens": tokens, "mismatches": mismatches,
                                     "bytes_per_token": round(total_bytes / tokens, 3), "files": rows}
        print(f"{name}: {tokens:,} tokens over {total_bytes / 1e6:.1f} MB, {mismatches} mismatches\n")

    report["measured_at"] = time.strftime("%Y-%m-%d")
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "parity.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 1 if any(e["mismatches"] for e in report["encodings"].values()) else 0


if __name__ == "__main__":
    sys.exit(main())
