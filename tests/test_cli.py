import json

from bytepair.cli import main


def test_train_encode_decode(tmp_path, capsys):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("the quick brown fox jumps over the lazy dog\n" * 40, encoding="utf-8")
    prefix = str(tmp_path / "tok")

    main(["train", str(corpus), "--vocab-size", "300", "--out", prefix])
    assert "bytes/token" in capsys.readouterr().out
    assert (tmp_path / "tok.model").exists() and (tmp_path / "tok.vocab").exists()

    main(["encode", "--model", prefix + ".model", "--text", "the lazy fox 🦊"])
    ids = json.loads(capsys.readouterr().out)
    main(["decode", "--model", prefix + ".model", "--ids", json.dumps(ids)])
    assert capsys.readouterr().out == "the lazy fox 🦊"


def test_encode_with_tiktoken_name(capsys):
    main(["encode", "--model", "cl100k_base", "--text", "hello world"])
    assert json.loads(capsys.readouterr().out) == [15339, 1917]
