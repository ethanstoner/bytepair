"""Encode throughput and training time: bytepair vs tiktoken, HF tokenizers and minbpe.

Everything runs on one thread (RAYON_NUM_THREADS=1 for HF). minbpe is only
used as a timing baseline and is read from temp/minbpe (a scratch clone).
Writes docs/benchmark.json.
"""

import os

os.environ["RAYON_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json  # noqa: E402
import pathlib  # noqa: E402
import platform  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "temp" / "minbpe"))

import tiktoken  # noqa: E402
import tokenizers  # noqa: E402
from tokenizers import Regex, decoders, models, pre_tokenizers, trainers  # noqa: E402

from bytepair import GPT4_PATTERN, SplitTokenizer, TiktokenTokenizer  # noqa: E402

CORPUS = ROOT / "temp" / "corpus"
HF_REPOS = {"cl100k_base": "Xenova/gpt-4", "o200k_base": "Xenova/gpt-4o"}


def timed(fn, repeats):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), out


def cpu_name():
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
        return out or platform.processor()
    except OSError:
        return platform.processor()


def bench_encode(text, repeats):
    nbytes = len(text.encode("utf-8"))
    mbps = lambda s: round(nbytes / s / 1e6, 2)  # noqa: E731
    results = {}
    for name, repo in HF_REPOS.items():
        ref = tiktoken.get_encoding(name)
        hf = tokenizers.Tokenizer.from_pretrained(repo)
        expected = ref.encode_ordinary(text)
        row = {}

        s, _ = timed(lambda: ref.encode_ordinary(text), repeats)
        row["tiktoken"] = mbps(s)
        # HF is slow on one huge string; a batch of lines is its fastest single-thread path.
        lines = text.splitlines(keepends=True)
        s, _ = timed(lambda: hf.encode_batch(lines, add_special_tokens=False), repeats)
        row["hf_tokenizers_batch_of_lines"] = mbps(s)

        ours = TiktokenTokenizer(name)

        def cold():
            ours._cache = {}
            return ours.encode_ordinary(text)

        s, ids = timed(cold, repeats)
        assert ids == expected, f"{name}: bytepair output differs from tiktoken"
        row["bytepair_cold_cache"] = mbps(s)
        s, _ = timed(lambda: ours.encode_ordinary(text), repeats)
        row["bytepair_warm_cache"] = mbps(s)

        # The textbook per-chunk loop, for the before/after of the rank-array encoder.
        fast = ours._encode_chunk
        ours._encode_chunk = ours._encode_chunk_simple
        s, ids = timed(cold, repeats)
        ours._encode_chunk = fast
        assert ids == expected
        row["bytepair_textbook_loop_cold"] = mbps(s)

        results[name] = row
        print(name, row, flush=True)
    return results


def hf_train(text, vocab_size):
    tok = tokenizers.Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(GPT4_PATTERN), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False, min_frequency=0,
    )
    tok.train_from_iterator([text], trainer)
    return tok


def bench_train(text, vocab_size, include_slow):
    nbytes = len(text.encode("utf-8"))
    row = {"train_bytes": nbytes, "vocab_size": vocab_size}

    s, tok = timed(lambda: SplitTokenizer().train(text, vocab_size, trainer="incremental"), 1)
    row["bytepair_incremental_s"] = round(s, 2)
    row["bytepair_bytes_per_token"] = round(nbytes / len(tok.encode(text, allowed_special="none")), 3)

    s, hf = timed(lambda: hf_train(text, vocab_size), 1)
    row["hf_tokenizers_s"] = round(s, 2)
    row["hf_bytes_per_token"] = round(nbytes / len(hf.encode(text).ids), 3)

    if include_slow:
        s, naive_tok = timed(lambda: SplitTokenizer().train(text, vocab_size, trainer="naive"), 1)
        row["bytepair_naive_s"] = round(s, 2)
        assert naive_tok.merges == tok.merges, "naive and incremental trainers disagree"

        from minbpe import RegexTokenizer

        s, _ = timed(lambda: RegexTokenizer().train(text, vocab_size), 1)
        row["minbpe_s"] = round(s, 2)

    print(row, flush=True)
    return row


def main(runs=3):
    files = sorted(CORPUS.glob("*.txt"))
    if not files:
        sys.exit("no corpus; run scripts/fetch_corpus.py first")
    text = "".join(p.read_bytes().decode("utf-8") for p in files)
    pride = (CORPUS / "en_pride_and_prejudice.txt").read_bytes().decode("utf-8")

    report = {
        "measured_at": time.strftime("%Y-%m-%d"),
        "cpu": cpu_name(),
        "python": platform.python_version(),
        "tiktoken": tiktoken.__version__,
        "tokenizers": tokenizers.__version__,
        "corpus_bytes": len(text.encode("utf-8")),
        "runs": [],
    }
    # Whole-suite repeats, because single runs drift between sessions; quote ranges.
    for i in range(runs):
        print(f"--- run {i + 1}/{runs}", flush=True)
        report["runs"].append({
            "encode_mb_per_s": bench_encode(text, repeats=3),
            "train": [
                bench_train(pride, 1024, include_slow=True),
                bench_train(text, 8192, include_slow=False),
            ],
        })
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
