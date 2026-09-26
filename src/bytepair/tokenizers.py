"""Concrete tokenizers."""

from . import _backend
from .base import Tokenizer

# Split patterns from OpenAI's GPT-2 and cl100k_base (GPT-4) tokenizers.
GPT2_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"""

class ByteTokenizer(Tokenizer):
    """BPE straight over the UTF-8 bytes of the whole text."""

    def __init__(self, backend="auto"):
        super().__init__(pattern=None, backend=backend)


class SplitTokenizer(Tokenizer):
    """Regex-split the text first so merges never cross word/number/space boundaries."""

    def __init__(self, pattern=GPT4_PATTERN, backend="auto"):
        super().__init__(pattern=pattern, backend=backend)


def _bpe_parts(ranks, token, max_rank):
    """Split `token` with BPE, using only merges ranked below `max_rank`."""
    parts = [bytes([b]) for b in token]
    while True:
        best_i, best_rank = None, None
        for i in range(len(parts) - 1):
            rank = ranks.get(parts[i] + parts[i + 1])
            if rank is not None and rank < max_rank and (best_rank is None or rank < best_rank):
                best_i, best_rank = i, rank
        if best_i is None:
            return parts
        parts[best_i : best_i + 2] = [parts[best_i] + parts[best_i + 1]]


class TiktokenTokenizer(SplitTokenizer):
    """An OpenAI encoding (cl100k_base, o200k_base, ...) rebuilt as an ordered merge list.

    tiktoken ships only {token bytes: rank}. Each multi-byte token was formed by
    merging two lower-ranked tokens, and running BPE on it with merges capped
    below its own rank stops at exactly those two parents. The 256 single bytes
    have shuffled ranks, so raw bytes are mapped through that permutation first.
    """

    def __init__(self, name, backend="auto"):
        import tiktoken

        enc = tiktoken.get_encoding(name)
        super().__init__(enc._pat_str, backend)
        self.name = name
        ranks = enc._mergeable_ranks
        self._ranks = ranks
        self.vocab = {rank: token for token, rank in ranks.items()}
        self.register_special_tokens(enc._special_tokens)
        if self.backend == "rust":
            return  # the extension rebuilds the merge tree itself, in parallel
        merges = {}
        for token, rank in ranks.items():
            if len(token) == 1:
                continue
            parts = _bpe_parts(ranks, token, rank)
            if len(parts) != 2:
                raise ValueError(f"{name}: token {token!r} has no two-part parent")
            merges[(ranks[parts[0]], ranks[parts[1]])] = rank
        self.merges = dict(sorted(merges.items(), key=lambda kv: kv[1]))
        self._byte_rank = [ranks[bytes([i])] for i in range(256)]

    def _make_rust(self):
        return _backend._core.Encoder.from_ranks(
            list(self._ranks.items()), self.pattern, list(self.special_tokens.items())
        )

    def _byte_ids(self, raw):
        return [self._byte_rank[b] for b in raw]

    def save(self, prefix):
        raise NotImplementedError("rebuilt from tiktoken; nothing to save")


class Cl100kTokenizer(TiktokenTokenizer):
    def __init__(self, backend="auto"):
        super().__init__("cl100k_base", backend)


class O200kTokenizer(TiktokenTokenizer):
    def __init__(self, backend="auto"):
        super().__init__("o200k_base", backend)
