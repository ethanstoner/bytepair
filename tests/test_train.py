import random

import pytest

from bytepair import GPT2_PATTERN, ByteTokenizer, SplitTokenizer, Tokenizer
from bytepair.trainers import incremental, naive

SAMPLES = [
    "",
    "hello world",
    "Hello, World! It's 2026 and we're tokenizing 12345678 things.",
    "ünïcödé naïve café, straße, İstanbul",
    "你好，世界。こんにちは世界。안녕하세요",
    "emoji 👨‍👩‍👧 👍🏽 🇺🇸 ❤️ 🤖",
    "tabs\tand   spaces \r\n and\rreturns   \n\n",
    "<|endoftext|> looks special but is plain text here",
]


def test_wikipedia_example():
    # https://en.wikipedia.org/wiki/Byte_pair_encoding: "aaabdaaabac" -> "XdXac"
    tok = ByteTokenizer().train("aaabdaaabac", 256 + 3)
    ids = tok.encode("aaabdaaabac")
    assert ids == [258, 100, 258, 97, 99]
    assert tok.decode(ids) == "aaabdaaabac"


@pytest.mark.parametrize("make", [ByteTokenizer, SplitTokenizer, lambda: SplitTokenizer(GPT2_PATTERN)])
def test_round_trip(make, sample_text):
    tok = make().train(sample_text, 256 + 200)
    for text in SAMPLES + [sample_text[:20_000]]:
        assert tok.decode(tok.encode(text, allowed_special="none")) == text


def test_training_compresses(sample_text):
    tok = SplitTokenizer().train(sample_text, 512)
    ids = tok.encode(sample_text)
    assert len(ids) < 0.6 * len(sample_text.encode("utf-8"))


def test_trainers_learn_identical_merges(sample_text):
    tok = SplitTokenizer()
    chunks = {}
    for piece in tok._split.findall(sample_text[:60_000]):
        key = tuple(piece.encode("utf-8"))
        chunks[key] = chunks.get(key, 0) + 1
    assert naive(chunks, 300) == incremental(chunks, 300)


def test_trainers_agree_on_random_bytes():
    rng = random.Random(1)
    for _ in range(20):
        chunks = {
            tuple(rng.choice(b"abcd") for _ in range(rng.randint(1, 12))): rng.randint(1, 5)
            for _ in range(rng.randint(1, 40))
        }
        assert naive(chunks, 30) == incremental(chunks, 30)


def test_training_stops_when_nothing_left_to_merge():
    tok = ByteTokenizer().train("ab", 1000)
    assert len(tok.merges) == 1


def test_save_load_round_trip(tmp_path, sample_text):
    tok = SplitTokenizer().train(sample_text, 400)
    tok.register_special_tokens({"<|endoftext|>": 400})
    tok.save(tmp_path / "tok")
    loaded = Tokenizer.load(tmp_path / "tok.model")
    assert loaded.merges == tok.merges
    assert loaded.pattern == tok.pattern
    for text in SAMPLES:
        assert loaded.encode(text, allowed_special="all") == tok.encode(text, allowed_special="all")
    vocab = (tmp_path / "tok.vocab").read_text(encoding="utf-8")
    assert "<special <|endoftext|>> 400" in vocab


def test_save_load_byte_tokenizer(tmp_path):
    tok = ByteTokenizer().train("the cat sat on the mat " * 20, 270)
    tok.save(tmp_path / "b")
    loaded = Tokenizer.load(tmp_path / "b.model")
    assert isinstance(loaded, ByteTokenizer)
    assert loaded.encode("the mat") == tok.encode("the mat")


def test_special_token_modes():
    tok = SplitTokenizer().train("hello world " * 50, 280)
    tok.register_special_tokens({"<|eot|>": 1000})
    text = "hello<|eot|>world"
    assert 1000 in tok.encode(text, allowed_special="all")
    assert 1000 in tok.encode(text, allowed_special={"<|eot|>"})
    assert 1000 not in tok.encode(text, allowed_special="none")
    with pytest.raises(ValueError):
        tok.encode(text)
    assert tok.decode(tok.encode(text, allowed_special="all")) == text
