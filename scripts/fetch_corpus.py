"""Download a mixed-language public-domain corpus into temp/corpus/.

Books come from Project Gutenberg with the licence header/footer stripped;
code comes from the local Python standard library. Also writes the small
committed sample at data/sample.txt.
"""

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "temp" / "corpus"

BOOKS = {
    "en_pride_and_prejudice": 1342,
    "en_moby_dick": 2701,
    "de_faust": 2229,
    "fr_les_miserables_1": 17489,
    "es_don_quijote": 2000,
    "zh_hongloumeng": 24264,
    "ja_rashomon": 1982,
    "ja_kessho_ki": 34013,
    "ja_kasei_no_kioku": 36358,
}


def strip_gutenberg(text):
    start = text.find("*** START OF")
    end = text.find("*** END OF")
    if start != -1:
        text = text[text.index("\n", start) + 1 :]
        end = text.find("*** END OF")
    if end != -1:
        text = text[:end]
    return text.strip() + "\n"


def fetch(book_id):
    # curl rather than urllib: urllib keeps truncating the larger books here.
    url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    raw = subprocess.run(
        ["curl", "-sSfL", "--retry", "5", "-A", "bytepair-corpus/1.0", url],
        check=True,
        capture_output=True,
    ).stdout
    return raw.decode("utf-8", errors="replace")


EDGE_POOL = (
    list("abcXYZ019 '\t\n\r.,!?-_()[]{}<>|/\\\"#$%^&*=+~`@")
    + [" ", " ", "　", "​", "﻿", "\r\n", "  ", "   \n"]
    + ["'s", "'LL", "'Ve", "'RE", "DON'T", "i'M"]
    + ["1234567", "3.14159", "0x7f", "1,000,000"]
    + ["\U0001f600", "\U0001f468‍\U0001f469‍\U0001f467", "\U0001f44d\U0001f3fd",
       "\U0001f1fa\U0001f1f8", "❤️", "\U0001f916"]
    + ["é", "é", "ß", "İ", "Αβ", "ж", "שלום",
       "مرحبا", "नमस्ते", "한국어",
       "你好", "こんにちは", "สวัสดี"]
    + ["<|endoftext|>", "<|fim_prefix|>", "<|im_start|>"]
)


def edge_cases(n_lines=20_000, seed=0):
    """Random mixtures of the characters and strings tokenizers tend to trip on."""
    import random

    rng = random.Random(seed)
    lines = []
    for _ in range(n_lines):
        lines.append("".join(rng.choice(EDGE_POOL) for _ in range(rng.randint(1, 40))))
    return "\n".join(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, book_id in BOOKS.items():
        path = OUT / f"{name}.txt"
        if path.exists():
            continue
        print(f"fetching {name} (#{book_id})", flush=True)
        path.write_text(strip_gutenberg(fetch(book_id)), encoding="utf-8", newline="")

    stdlib = pathlib.Path(sys.base_prefix) / "Lib"
    code = [p.read_text(encoding="utf-8", errors="replace") for p in sorted(stdlib.glob("*.py"))[:60]]
    (OUT / "code_python_stdlib.txt").write_text("\n".join(code), encoding="utf-8", newline="")

    (OUT / "edge_cases.txt").write_text(edge_cases(), encoding="utf-8", newline="")

    pride = (OUT / "en_pride_and_prejudice.txt").read_bytes().decode("utf-8")
    sample = ROOT / "data" / "sample.txt"
    sample.parent.mkdir(exist_ok=True)
    sample.write_text(pride[:150_000], encoding="utf-8", newline="")

    for p in sorted(OUT.glob("*.txt")):
        print(f"{p.name:32} {p.stat().st_size / 1e6:6.2f} MB")


if __name__ == "__main__":
    main()
