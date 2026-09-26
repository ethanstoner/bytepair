"""Command line: train, encode, decode."""

import argparse
import json
import pathlib
import sys
import time

from .base import Tokenizer
from .tokenizers import GPT2_PATTERN, GPT4_PATTERN, ByteTokenizer, SplitTokenizer, TiktokenTokenizer

PATTERNS = {"gpt4": GPT4_PATTERN, "gpt2": GPT2_PATTERN}


def load_tokenizer(spec):
    """A saved `.model` path, or the name of a tiktoken encoding (cl100k_base, o200k_base)."""
    if spec.endswith(".model"):
        return Tokenizer.load(spec)
    return TiktokenTokenizer(spec)


def read_text(args):
    if args.file:
        return pathlib.Path(args.file).read_bytes().decode("utf-8")
    if args.text is not None:
        return args.text
    return sys.stdin.read()


def cmd_train(args):
    text = pathlib.Path(args.input).read_bytes().decode("utf-8")
    tok = ByteTokenizer() if args.pattern == "none" else SplitTokenizer(PATTERNS[args.pattern])
    t0 = time.perf_counter()
    tok.train(text, args.vocab_size, trainer=args.trainer)
    elapsed = time.perf_counter() - t0
    tok.save(args.out)
    nbytes = len(text.encode("utf-8"))
    ntok = len(tok.encode(text, allowed_special="none"))
    print(f"trained {len(tok.merges)} merges in {elapsed:.2f}s -> {args.out}.model, {args.out}.vocab")
    print(f"{nbytes:,} bytes -> {ntok:,} tokens ({nbytes / ntok:.2f} bytes/token)")


def cmd_encode(args):
    tok = load_tokenizer(args.model)
    ids = tok.encode(read_text(args), allowed_special="all" if args.allow_special else "none_raise")
    print(json.dumps(ids))


def cmd_decode(args):
    tok = load_tokenizer(args.model)
    ids = json.loads(args.ids if args.ids is not None else sys.stdin.read())
    sys.stdout.write(tok.decode(ids))


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="bytepair")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="learn merges from a text file")
    p.add_argument("input")
    p.add_argument("--vocab-size", type=int, default=512)
    p.add_argument("--pattern", choices=["gpt4", "gpt2", "none"], default="gpt4")
    p.add_argument("--trainer", choices=["incremental", "naive"], default="incremental")
    p.add_argument("--out", default="tokenizer")
    p.set_defaults(func=cmd_train)

    for name, func in (("encode", cmd_encode), ("decode", cmd_decode)):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True, help="path to a .model file, or cl100k_base / o200k_base")
        p.set_defaults(func=func)
    enc, dec = sub.choices["encode"], sub.choices["decode"]
    enc.add_argument("--text")
    enc.add_argument("--file")
    enc.add_argument("--allow-special", action="store_true")
    dec.add_argument("--ids", help="JSON list of ids (default: stdin)")

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
