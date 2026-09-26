"""The Rust extension must agree with the Python reference and with tiktoken."""

import pytest
import regex
import tiktoken

from bytepair import GPT2_PATTERN, GPT4_PATTERN, _backend

core = _backend._core
pytestmark = pytest.mark.skipif(core is None, reason="bytepair._core not built")

O200K_PATTERN = tiktoken.get_encoding("o200k_base")._pat_str

SPLIT_CASES = [
    "",
    "hello world",
    "I'M don't THEY'LL we'Ve it'ſ",
    "HELLOWorld helloWORLD ǅemo ǄEMO ʰa中文ʰA",
    "éclair ́x 1́",
    "numbers 1 12 123 1234 12345 ٣٤٥٦",
    "path/to//file a//\n",
    "   leading and trailing   ",
    "multiple\n\n\n\nnewlines \r\n crlf \r lone cr  \n \n  x",
    " nbsp　　ideographic​zero-width﻿bom",
    "emoji 👨‍👩‍👧 👍🏽 🇺🇸 ❤️ 🤖",
    "你好，世界。こんにちは世界。안녕하세요 مرحبا नमस्ते",
]


def test_known_patterns_match_the_real_ones():
    assert core.CL100K_PATTERN == GPT4_PATTERN == tiktoken.get_encoding("cl100k_base")._pat_str
    assert core.O200K_PATTERN == O200K_PATTERN


@pytest.mark.parametrize("pattern", [GPT4_PATTERN, O200K_PATTERN, GPT2_PATTERN])
@pytest.mark.parametrize("force_regex", [False, True])
def test_split_matches_findall(pattern, force_regex, sample_text):
    for text in SPLIT_CASES + [sample_text]:
        assert core.split(pattern, text, force_regex) == regex.findall(pattern, text)


def test_bad_pattern_raises_value_error():
    with pytest.raises(ValueError):
        core.split("(unclosed", "x")


def _trained(backend, sample_text):
    from bytepair import SplitTokenizer

    tok = SplitTokenizer(backend=backend).train(sample_text[:40_000], 400)
    tok.register_special_tokens({"<|a|>": 1000, "<|a|>b": 1001, "<|end|>": 1002})
    return tok


def test_rust_and_python_encode_identically(sample_text):
    py, rs = _trained("python", sample_text), _trained("rust", sample_text)
    for text in SPLIT_CASES + [sample_text, "x<|a|>b<|end|>  ", "<|a|>b<|a|>"]:
        for mode in ["all", "none", {"<|a|>b"}, {"<|end|>", "<|a|>b"}]:
            assert rs.encode(text, mode) == py.encode(text, mode), (text[:40], mode)
        assert rs.decode(rs.encode(text, "all")) == text


def test_specials_tie_break_on_registration_order(backend, sample_text):
    tok = _trained(backend, sample_text)
    # "<|a|>" was registered first, so it wins over "<|a|>b" at the same position,
    # whatever order the caller's set iterates in.
    assert tok.encode("<|a|>b", {"<|a|>b", "<|a|>"})[0] == 1000


@pytest.mark.parametrize(
    "call",
    [
        lambda t: t.encode("has <|end|> inside"),
        lambda t: t.encode("x", {"<|nope|>"}),
        lambda t: t.encode("x", "sometimes"),
        lambda t: t.decode([999_999]),
    ],
)
def test_errors_are_value_errors(backend, sample_text, call):
    with pytest.raises(ValueError):
        call(_trained(backend, sample_text))


def test_bad_pattern_is_value_error(backend):
    from bytepair import SplitTokenizer

    with pytest.raises(ValueError):
        SplitTokenizer("(unclosed", backend)


@pytest.mark.parametrize("raw", [b"\xe4\xbd", b"\xff", b"\x80abc", b"a\xe4\xbd\xa0\xe4", b"\xf0\x9f\x98"])
def test_lossy_decode_matches_python(raw):
    from bytepair import ByteTokenizer

    ids = list(raw)  # an untrained byte tokenizer: id == byte
    for backend in ["python", "rust"]:
        assert ByteTokenizer(backend).decode(ids) == raw.decode("utf-8", errors="replace")


def test_auto_falls_back_without_extension(monkeypatch):
    from bytepair import SplitTokenizer

    monkeypatch.setattr(_backend, "_core", None)
    assert SplitTokenizer().backend == "python"
    with pytest.raises(ImportError):
        SplitTokenizer(backend="rust")


def test_auto_falls_back_on_pattern_rust_cannot_compile():
    from bytepair import SplitTokenizer

    pattern = r"\p{Script=Latin}+|\X"  # Python regex only: \X is not supported by fancy-regex
    with pytest.raises(ValueError):
        core.split(pattern, "abc")
    tok = SplitTokenizer(pattern).train("abc déf " * 20, 260)
    assert tok.decode(tok.encode("abc déf")) == "abc déf"
    assert tok.backend == "python"


def test_encode_batch_matches_single(sample_text):
    enc = core.Encoder.from_ranks(
        list(tiktoken.get_encoding("cl100k_base")._mergeable_ranks.items()), GPT4_PATTERN, []
    )
    lines = sample_text.splitlines(keepends=True)[:2000]
    ref = tiktoken.get_encoding("cl100k_base")
    assert enc.encode_batch(lines) == [ref.encode_ordinary(line) for line in lines]
