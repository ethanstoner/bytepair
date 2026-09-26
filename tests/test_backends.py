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
