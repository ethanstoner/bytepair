"""Picks between the Rust extension and the pure-Python reference implementation."""

try:
    from . import _core
except ImportError:  # extension not built
    _core = None

BACKENDS = ("auto", "rust", "python")


def resolve(backend):
    if backend == "python":
        return "python"
    if backend == "rust":
        if _core is None:
            raise ImportError("bytepair._core is not built; run `maturin develop --release`")
        return "rust"
    if backend == "auto":
        return "rust" if _core is not None else "python"
    raise ValueError(f"unknown backend {backend!r}; expected one of {BACKENDS}")
