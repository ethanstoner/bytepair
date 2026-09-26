"""Shared encode/decode, special-token handling and save/load."""

import json

import regex

from .core import render_token, replace_pair
from .trainers import TRAINERS

FORMAT = "bytepair v1"


class Tokenizer:
    def __init__(self, pattern=None):
        self.pattern = pattern
        self._split = regex.compile(pattern) if pattern else None
        self.merges = {}  # (id, id) -> new id, in learned order
        self.special_tokens = {}  # str -> id
        self._special_ids = {}  # id -> str
        self.vocab = self._build_vocab()
        self._cache = {}

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

    def register_special_tokens(self, specials):
        self.special_tokens = dict(specials)
        self._special_ids = {i: s for s, i in self.special_tokens.items()}

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

    def _encode_chunk(self, raw):
        ids = self._byte_ids(raw)
        merges = self.merges
        while len(ids) >= 2:
            # Apply the earliest-learned merge present; later merges may depend on it.
            pair = min(zip(ids, ids[1:]), key=lambda p: merges.get(p, float("inf")))
            if pair not in merges:
                break
            ids = replace_pair(ids, pair, merges[pair])
        return ids

    def encode_ordinary(self, text):
        """Encode text, treating special-token strings as plain text."""
        chunks = self._split.findall(text) if self._split else [text]
        out = []
        cache = self._cache
        for chunk in chunks:
            ids = cache.get(chunk)
            if ids is None:
                ids = self._encode_chunk(chunk.encode("utf-8"))
                if len(cache) < 200_000:
                    cache[chunk] = ids
            out.extend(ids)
        return out

    def encode(self, text, allowed_special="none_raise"):
        """Encode text.

        allowed_special: "all", "none" (specials encoded as plain text),
        "none_raise" (error if any special string appears, tiktoken's default)
        or a set of special-token strings to honour.
        """
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
            allowed = {s: self.special_tokens[s] for s in allowed_special}
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
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            header = f.readline().rstrip("\n")
            if header != FORMAT:
                raise ValueError(f"{path}: expected {FORMAT!r}, got {header!r}")
            pattern = json.loads(f.readline())
            specials = json.loads(f.readline())
            merge_list = [tuple(map(int, line.split())) for line in f if line.strip()]
        from .tokenizers import ByteTokenizer, SplitTokenizer

        tok = ByteTokenizer() if pattern is None else SplitTokenizer(pattern)
        tok._set_merges(merge_list)
        tok.register_special_tokens(specials)
        return tok
