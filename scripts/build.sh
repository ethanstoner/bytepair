#!/usr/bin/env bash
# Build bytepair._core into the project venv. On Windows the MSVC toolchain is not
# required: the GNU target links against CPython via pyo3's generate-import-lib.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "${OS:-}" == "Windows_NT" ]]; then
  export RUSTUP_TOOLCHAIN=stable-x86_64-pc-windows-gnu
  export VIRTUAL_ENV="$PWD/venv"
  exec venv/Scripts/maturin develop --release --target x86_64-pc-windows-gnu "$@"
fi
exec maturin develop --release "$@"
