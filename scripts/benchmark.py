"""Encode throughput and training time: bytepair (Rust and Python) vs tiktoken, HF
tokenizers and minbpe.

Thread pools read their size once per process, so each run is two subprocesses:
  single  RAYON_NUM_THREADS=1   every single-thread row
  multi   RAYON_NUM_THREADS=N   batch encoding and parallel training, N = all logical cores
The parent repeats the pair `runs` times and writes every run to docs/benchmark.json;
the README quotes min-max ranges across runs. Every bytepair encode is checked against
tiktoken before its time counts. minbpe is read from temp/minbpe (a scratch clone) and is
only a timing baseline. Run with nothing else busy on the machine.

    python scripts/benchmark.py [runs]
"""

import json
import os
import pathlib
import platform
import statistics
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
CORPUS = ROOT / "temp" / "corpus"
HF_REPOS = {"cl100k_base": "Xenova/gpt-4", "o200k_base": "Xenova/gpt-4o"}
ENCODINGS = ("cl100k_base", "o200k_base")


def timed(fn, repeats):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), out


def load_corpus():
    files = sorted(CORPUS.glob("*.txt"))
    if not files:
        sys.exit("no corpus; run scripts/fetch_corpus.py first")
    text = "".join(p.read_bytes().decode("utf-8") for p in files)
    pride = (CORPUS / "en_pride_and_prejudice.txt").read_bytes().decode("utf-8")
    return text, pride


def documents(text, size=64_000):
    """Cut the corpus into ~64 KB documents at line boundaries: the batch unit for the
    multi-thread phase. (tiktoken's batch encoder schedules one thread-pool task per
    item, so feeding it single lines would measure task overhead, not encoding.)"""
    docs, cur, n = [], [], 0
    for line in text.splitlines(keepends=True):
        cur.append(line)
        n += len(line)
        if n >= size:
            docs.append("".join(cur))
            cur, n = [], 0
    if cur:
        docs.append("".join(cur))
    return docs


def hf_train(text, vocab_size):
    import tokenizers
    from tokenizers import Regex, decoders, models, pre_tokenizers, trainers

    from bytepair import GPT4_PATTERN

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


def phase_single(repeats=3):
    import tiktoken
    import tokenizers

    from bytepair import SplitTokenizer, TiktokenTokenizer

    text, pride = load_corpus()
    nbytes = len(text.encode("utf-8"))
    lines = text.splitlines(keepends=True)
    mbps = lambda s: round(nbytes / s / 1e6, 2)  # noqa: E731
    encode = {}
    for name in ENCODINGS:
        ref = tiktoken.get_encoding(name)
        expected = ref.encode_ordinary(text)
        row = {}
        s, _ = timed(lambda: ref.encode_ordinary(text), repeats)
        row["tiktoken"] = mbps(s)
        hf = tokenizers.Tokenizer.from_pretrained(HF_REPOS[name])
        s, _ = timed(lambda: hf.encode_batch(lines, add_special_tokens=False), repeats)
        row["hf_tokenizers_batch_of_lines"] = mbps(s)

        for backend in ("rust", "python"):
            tok = TiktokenTokenizer(name, backend=backend)
            assert tok.backend == backend

            def cold():
                tok.clear_cache()
                return tok.encode_ordinary(text)

            s, ids = timed(cold, repeats)
            assert ids == expected, f"{name} {backend}: output differs from tiktoken"
            row[f"bytepair_{backend}_cold"] = mbps(s)
            s, ids = timed(lambda: tok.encode_ordinary(text), repeats)
            assert ids == expected
            row[f"bytepair_{backend}_warm"] = mbps(s)
            if backend == "python":
                fast = tok._encode_chunk
                tok._encode_chunk = tok._encode_chunk_simple
                s, ids = timed(cold, repeats)
                tok._encode_chunk = fast
                assert ids == expected
                row["bytepair_python_textbook_cold"] = mbps(s)
        encode[name] = row
        print(name, row, file=sys.stderr, flush=True)

    train = []
    for label, corpus, vocab in (("pride_1024", pride, 1024), ("corpus_8192", text, 8192)):
        cbytes = len(corpus.encode("utf-8"))
        row = {"name": label, "train_bytes": cbytes, "vocab_size": vocab}
        s, rust = timed(lambda: SplitTokenizer(backend="rust").train(corpus, vocab), 1)
        row["bytepair_rust_s"] = round(s, 3)
        row["bytepair_bytes_per_token"] = round(cbytes / len(rust.encode(corpus, "none")), 3)
        s, py = timed(lambda: SplitTokenizer(backend="python").train(corpus, vocab), 1)
        row["bytepair_python_s"] = round(s, 2)
        assert py.merges == rust.merges, f"{label}: rust and python trainers disagree"
        s, hf = timed(lambda: hf_train(corpus, vocab), 1)
        row["hf_tokenizers_s"] = round(s, 3)
        row["hf_bytes_per_token"] = round(cbytes / len(hf.encode(corpus).ids), 3)
        if label == "pride_1024":
            s, _ = timed(lambda: SplitTokenizer(backend="python").train(corpus, vocab, trainer="naive"), 1)
            row["bytepair_python_naive_s"] = round(s, 2)
            sys.path.insert(0, str(ROOT / "temp" / "minbpe"))
            from minbpe import RegexTokenizer

            s, _ = timed(lambda: RegexTokenizer().train(corpus, vocab), 1)
            row["minbpe_s"] = round(s, 2)
        train.append(row)
        print(row, file=sys.stderr, flush=True)
    return {"encode_mb_per_s": encode, "train": train}


def phase_multi(threads, repeats=3):
    import tiktoken
    import tokenizers

    from bytepair import SplitTokenizer, TiktokenTokenizer

    text, _ = load_corpus()
    nbytes = len(text.encode("utf-8"))
    docs = documents(text)
    mbps = lambda s: round(nbytes / s / 1e6, 2)  # noqa: E731
    encode = {"documents": len(docs)}
    for name in ENCODINGS:
        ref = tiktoken.get_encoding(name)
        expected = [ref.encode_ordinary(d) for d in docs]
        row = {}
        s, _ = timed(lambda: ref.encode_ordinary_batch(docs, num_threads=threads), repeats)
        row["tiktoken_batch"] = mbps(s)
        hf = tokenizers.Tokenizer.from_pretrained(HF_REPOS[name])
        s, _ = timed(lambda: hf.encode_batch(docs, add_special_tokens=False), repeats)
        row["hf_tokenizers_batch"] = mbps(s)
        tok = TiktokenTokenizer(name, backend="rust")
        engine = tok._engine()
        s, ids = timed(lambda: engine.encode_batch(docs), repeats)
        assert ids == expected, f"{name}: rust batch output differs from tiktoken"
        row["bytepair_rust_batch"] = mbps(s)
        encode[name] = row
        print(name, row, file=sys.stderr, flush=True)

    s, tok = timed(lambda: SplitTokenizer(backend="rust").train(text, 8192), 1)
    train = {"name": "corpus_8192", "bytepair_rust_parallel_s": round(s, 3)}
    s, _ = timed(lambda: hf_train(text, 8192), 1)
    train["hf_tokenizers_parallel_s"] = round(s, 3)
    print(train, file=sys.stderr, flush=True)
    return {"threads": threads, "encode_mb_per_s": encode, "train": train}


def cpu_name():
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
        return out or platform.processor()
    except OSError:
        return platform.processor()


def run_phase(phase, threads):
    env = dict(os.environ, RAYON_NUM_THREADS=str(threads), TOKENIZERS_PARALLELISM="true")
    out = subprocess.run(
        [sys.executable, __file__, "--phase", phase, str(threads)],
        env=env, stdout=subprocess.PIPE, check=True, text=True,
    ).stdout
    return json.loads(out.strip().splitlines()[-1])


def main(runs):
    import tiktoken
    import tokenizers

    from bytepair import _backend

    threads = os.cpu_count()
    report = {
        "measured_at": time.strftime("%Y-%m-%d"),
        "cpu": cpu_name(),
        "logical_cores": threads,
        "python": platform.python_version(),
        "tiktoken": tiktoken.__version__,
        "tokenizers": tokenizers.__version__,
        "bytepair_core": _backend._core.__version__,
        "corpus_bytes": len(load_corpus()[0].encode("utf-8")),
        "runs": [],
    }
    for i in range(runs):
        print(f"--- run {i + 1}/{runs}", file=sys.stderr, flush=True)
        report["runs"].append({"single": run_phase("single", 1), "multi": run_phase("multi", threads)})
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    if len(sys.argv) > 1 and sys.argv[1] == "--phase":
        result = phase_single() if sys.argv[2] == "single" else phase_multi(int(sys.argv[3]))
        print(json.dumps(result))
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
