import pathlib

import pytest

from bytepair import _backend

SAMPLE = pathlib.Path(__file__).resolve().parents[1] / "data" / "sample.txt"


@pytest.fixture(scope="session")
def sample_text():
    return SAMPLE.read_bytes().decode("utf-8")


@pytest.fixture(params=["python", "rust"])
def backend(request):
    if request.param == "rust" and _backend._core is None:
        pytest.skip("bytepair._core not built")
    return request.param
