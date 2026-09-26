"""Our rebuilt OpenAI encodings must match tiktoken token for token."""

import pytest
import tiktoken

from bytepair import Cl100kTokenizer, O200kTokenizer, _backend

CASES = [
    "",
    " ",
    "\n",
    "hello world",
    "Hello world!!! I'M DON'T we'Ve they'RE",
    "   leading and trailing   ",
    "trailing newline\n",
    "multiple\n\n\n\nnewlines \r\n crlf \r lone cr",
    "numbers 1 12 123 1234 12345 3.14159 1,000,000 0x7fff",
    "ünïcödé naïve café straße İstanbul é",
    "你好，世界。こんにちは世界。안녕하세요 مرحبا नमस्ते สวัสดี",
    "emoji 👨‍👩‍👧 👍🏽 🇺🇸 ❤️ 🤖🤖🤖",
    " nbsp　ideographic​zero-width﻿bom",
    "def f(x):\n\treturn {'a': [1, 2, 3]}  # comment\n",
    "a" * 5000,
    " " * 300 + "x",
]


@pytest.fixture(
    scope="module",
    params=[
        (cls, name, backend)
        for cls, name in [(Cl100kTokenizer, "cl100k_base"), (O200kTokenizer, "o200k_base")]
        for backend in ["python", "rust"]
    ],
    ids=lambda p: f"{p[1]}-{p[2]}",
)
def pair(request):
    cls, name, backend = request.param
    if backend == "rust" and _backend._core is None:
        pytest.skip("bytepair._core not built")
    return cls(backend), tiktoken.get_encoding(name)


@pytest.mark.parametrize("text", CASES)
def test_encode_matches(pair, text):
    ours, ref = pair
    assert ours.encode(text) == ref.encode(text)
    assert ours.decode(ours.encode(text)) == text


def test_specials_match(pair):
    ours, ref = pair
    text = "a<|endoftext|>b<|endofprompt|>c"
    assert ours.encode(text, allowed_special="all") == ref.encode(text, allowed_special="all")
    assert ours.encode(text, allowed_special="none") == ref.encode(text, disallowed_special=())
    with pytest.raises(ValueError):
        ours.encode(text)


def test_sample_file_matches(pair, sample_text):
    ours, ref = pair
    assert ours.encode(sample_text) == ref.encode(sample_text)
