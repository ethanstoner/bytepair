"""Shared encode/decode, special-token handling and save/load."""

import json

import regex

from . import _backend
from .core import render_token, replace_pair
from .trainers import TRAINERS

FORMAT = "bytepair v1"
CACHE_LIMIT = 1_000_000  # distinct pre-split chunks remembered per tokenizer
_INF = float("inf")


class Tokenizer:
    """backend: "rust" (the bytepair._core extension), "python" (this file, the readable
    reference) or "auto" (rust when the extension is built, else python). Both give
    identical output."""

    def __init__(self, pattern=None, backend="auto"):
        self.pattern = pattern
        try:
            self._split = regex.compile(pattern) if pattern else None
        except regex.error as e:
            raise ValueError(f"bad pattern: {e}") from None
        self._requested_backend = backend
        self.backend = _backend.resolve(backend)
        self._rust = None
        self.merges = {}  # (id, id) -> new id, in learned order
        self.special_tokens = {}  # str -> id
        self._special_ids = {}  # id -> str
        self.vocab = self._build_vocab()
        self._cache = {}

    # ---- backend ----------------------------------------------------------

    def _make_rust(self):
        return _backend._core.Encoder.from_merges(
            list(self.merges), self.pattern, list(self.special_tokens.items())
        )

    def _engine(self):
        """The Rust encoder, built on first use; None when running the Python backend."""
        if self.backend != "rust":
            return None
        if self._rust is None:
            try:
                self._rust = self._make_rust()
            except ValueError:
                # e.g. a custom pattern the Rust regex engine cannot compile
                if self._requested_backend != "auto":
                    raise
                self.backend = "python"
                return None
        return self._rust

    def clear_cache(self):
        self._cache = {}
        if self._rust is not None:
            self._rust.clear_cache()

    # ---- vocabulary -------------------------------------------------------

    def _build_vocab(self):
        vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), new_id in self.merges.items():
            vocab[new_id] = vocab[a] + vocab[b]
        return vocab

    def _set_merges(self, merge_list, first_id=256):
        self.merges = {pair: first_id + i for i, pair in enumerate(merge_list)}
        self.vocab = self._build_vocab()
        self._cache = {}
        self._rust = None

    def register_special_tokens(self, specials):
        self.special_tokens = dict(specials)
        self._special_ids = {i: s for s, i in self.special_tokens.items()}
        self._rust = None

    @property
    def vocab_size(self):
        return len(self.vocab) + len(self.special_tokens)

    # ---- training ---------------------------------------------------------

    def train(self, text, vocab_size, trainer="incremental", verbose=False):
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256")
        pieces = self._split.findall(text) if self._split else [text]
        chunks = {}
        for piece in pieces:
            key = tuple(piece.encode("utf-8"))
            chunks[key] = chunks.get(key, 0) + 1
        merge_list = TRAINERS[trainer](chunks, vocab_size - 256)
        self._set_merges(merge_list)
        if verbose:
            for pair, new_id in self.merges.items():
                print(f"merge {pair} -> {new_id} [{render_token(self.vocab[new_id])}]")
        return self

    # ---- encoding ---------------------------------------------------------

    def _byte_ids(self, raw):
        return list(raw)

    def _encode_chunk_simple(self, raw):
        """Textbook version: find the earliest-learned pair present, replace it everywhere, repeat."""
        ids = self._byte_ids(raw)
        merges = self.merges
        while len(ids) >= 2:
            pair = min(zip(ids, ids[1:]), key=lambda p: merges.get(p, float("inf")))
            if pair not in merges:
                break
            ids = replace_pair(ids, pair, merges[pair])
        return ids

    def _encode_chunk(self, raw):
        """Same result as _encode_chunk_simple, without rescanning every pair per merge.

        Keeps the rank of each adjacent pair in a list, merges the leftmost
        lowest-ranked pair and re-ranks only its two neighbours. A merge id is
        its rank, and any pair containing the new id ranks after it, so taking
        occurrences one at a time left to right matches replacing them all at once.
        """
        ids = self._byte_ids(raw)
        if len(ids) < 2:
            return ids
        get = self.merges.get
        inf = _INF
        ranks = [get(p, inf) for p in zip(ids, ids[1:])]
        while ranks:
            best = min(ranks)
            if best == inf:
                break
            i = ranks.index(best)
            ids[i : i + 2] = [best]
            del ranks[i]
            if i > 0:
                ranks[i - 1] = get((ids[i - 1], best), inf)
            if i < len(ranks):
                ranks[i] = get((best, ids[i + 1]), inf)
        return ids

    def encode_ordinary(self, text):
        """Encode text, treating special-token strings as plain text."""
        rust = self._engine()
        if rust is not None:
            return rust.encode_ordinary(text)
        chunks = self._split.findall(text) if self._split else [text]
        out = []
        cache = self._cache
        for chunk in chunks:
            ids = cache.get(chunk)
            if ids is None:
                ids = self._encode_chunk(chunk.encode("utf-8"))
                if len(cache) < CACHE_LIMIT:
                    cache[chunk] = ids
            out.extend(ids)
        return out

    def encode(self, text, allowed_special="none_raise"):
        """Encode text.

        allowed_special: "all", "none" (specials encoded as plain text),
        "none_raise" (error if any special string appears, tiktoken's default)
        or a set of special-token strings to honour.
        """
        rust = self._engine()
        if rust is not None:
            if isinstance(allowed_special, (set, frozenset, list, tuple)):
                allowed_special = list(allowed_special)
            return rust.encode(text, allowed_special)
        if allowed_special == "all":
            allowed = self.special_tokens
        elif allowed_special == "none":
            allowed = {}
        elif allowed_special == "none_raise":
            for s in self.special_tokens:
                if s in text:
                    raise ValueError(f"text contains special token {s!r}")
            allowed = {}
        elif isinstance(allowed_special, (set, frozenset, list, tuple)):
            unknown = [s for s in allowed_special if s not in self.special_tokens]
            if unknown:
                raise ValueError(f"unknown special token {unknown[0]!r}")
            # Registration order, not the caller's set order, decides ties between specials.
            allowed = {s: i for s, i in self.special_tokens.items() if s in allowed_special}
        else:
            raise ValueError(f"bad allowed_special: {allowed_special!r}")

        if not allowed:
            return self.encode_ordinary(text)
        splitter = "(" + "|".join(regex.escape(s) for s in allowed) + ")"
        out = []
        for part in regex.split(splitter, text):
            if part in allowed:
                out.append(allowed[part])
            elif part:
                out.extend(self.encode_ordinary(part))
        return out

    # ---- decoding ---------------------------------------------------------

    def decode(self, ids):
        rust = self._engine()
        if rust is not None:
            return rust.decode(list(ids))
        specials = self._special_ids
        parts = []
        for i in ids:
            if i in self.vocab:
                parts.append(self.vocab[i])
            elif i in specials:
                parts.append(specials[i].encode("utf-8"))
            else:
                raise ValueError(f"unknown token id {i}")
        return b"".join(parts).decode("utf-8", errors="replace")

    # ---- persistence ------------------------------------------------------

    def save(self, prefix):
        """Write `prefix.model` (loadable) and `prefix.vocab` (for humans)."""
        with open(f"{prefix}.model", "w", encoding="utf-8") as f:
            f.write(FORMAT + "\n")
            f.write(json.dumps(self.pattern) + "\n")
            f.write(json.dumps(self.special_tokens, ensure_ascii=False) + "\n")
            for a, b in self.merges:
                f.write(f"{a} {b}\n")

        parents = {new_id: pair for pair, new_id in self.merges.items()}
        with open(f"{prefix}.vocab", "w", encoding="utf-8") as f:
            for i, raw in self.vocab.items():
                if i in parents:
                    a, b = parents[i]
                    f.write(
                        f"[{render_token(self.vocab[a])}][{render_token(self.vocab[b])}]"
                        f" -> [{render_token(raw)}] {i}\n"
                    )
                else:
                    f.write(f"[{render_token(raw)}] {i}\n")
            for s, i in self.special_tokens.items():
                f.write(f"<special {s}> {i}\n")

    @classmethod
    def load(cls, path, backend="auto"):
        with open(path, encoding="utf-8") as f:
            header = f.readline().rstrip("\n")
            if header != FORMAT:
                raise ValueError(f"{path}: expected {FORMAT!r}, got {header!r}")
            pattern = json.loads(f.readline())
            specials = json.loads(f.readline())
            merge_list = [tuple(map(int, line.split())) for line in f if line.strip()]
        from .tokenizers import ByteTokenizer, SplitTokenizer

        tok = ByteTokenizer(backend) if pattern is None else SplitTokenizer(pattern, backend)
        tok._set_merges(merge_list)
        tok.register_special_tokens(specials)
        return tok
