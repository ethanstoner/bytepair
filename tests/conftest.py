import pathlib

import pytest

SAMPLE = pathlib.Path(__file__).resolve().parents[1] / "data" / "sample.txt"


@pytest.fixture(scope="session")
def sample_text():
    return SAMPLE.read_bytes().decode("utf-8")
