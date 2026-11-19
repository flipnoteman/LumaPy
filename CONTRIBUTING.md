# Contributing to LumaPy

LumaPy is a [PyO3](https://pyo3.rs) extension built with [maturin](https://www.maturin.rs), laid out like other
Rust-backed Python packages (pydantic-core, polars, orjson):

```
Cargo.toml, src/          the Rust extension module `lumapy._lumapy`
pyproject.toml            Python packaging (maturin build backend), ruff, mypy and pytest settings
python/lumapy/            the Python package: __init__.py (re-exports and NumPy-style functions),
                          _lumapy.pyi (type stubs of the extension), py.typed
tests/                    pytest suite, checked against NumPy (and scikit-learn)
examples/, benchmarks/    Python ports of Luma's examples; benchmarks against CPU libraries
.github/workflows/        ci.yml (lint and tests), release.yml (wheels and PyPI publishing)
```

## Setup

You need a Rust toolchain (1.99 or later) and Python 3.10 or later.

```sh
python -m venv .venv
. .venv/bin/activate                  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip   # pip >= 25.1 for --group
pip install --group dev
maturin develop --release             # builds the extension into the venv; rerun after Rust changes
pytest
```

`maturin develop` without `--release` builds faster but runs the GPU host code much slower.

## Checks

CI runs all of these; [pre-commit](https://pre-commit.com) runs the formatters and linters on commit
(`pre-commit install`).

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
ruff check python tests examples benchmarks
ruff format --check python tests examples benchmarks
mypy                                   # the Python package and the stubs
python -m mypy.stubtest lumapy._lumapy # the stubs against the built extension
pytest
```

`LUMA_DEVICE=cpu pytest` runs the tests on a software Vulkan driver, as CI does on Linux.

## Developing against a local Luma

`Cargo.toml` pins Luma to a commit on GitHub, so builds are reproducible anywhere. To build against a Luma checkout
next to this repository instead (`../Luma`), override the dependency without editing `Cargo.toml`, for example in a
`.cargo/config.toml` in the directory that holds both repositories (Cargo reads config files from parent
directories):

```toml
[patch."https://github.com/flipnoteman/Luma"]
luma = { path = "Luma" }
```

or for a single build:

```sh
cargo build --config 'patch."https://github.com/flipnoteman/Luma".luma.path="../Luma"'
```

When a change needs a new Luma commit, push it to Luma and update the `rev` in `Cargo.toml` (and `Cargo.lock` with
`cargo update -p luma`). Then update the "Mirrors Luma" line at the top of the README and its feature parity
table; `tests/test_docs.py` fails while the README names another version or commit than the build. Once Luma publishes a release with everything the bindings use, depend on the crates.io
version instead.

## Adding a binding

1. Add the method or function in the matching `src/` module (`ops.rs` operators and elementwise math, `reduce.rs`
   reductions, sorting and scatters, `view.rs` views and indexing, `functions.rs` module functions). Shapes and
   axes arrive in NumPy order; convert them with `inner::luma_shape` / `inner::luma_axis`.
2. Export new module-level functions and classes in the `#[pymodule]` in `src/lib.rs`, and re-export them from
   `python/lumapy/__init__.py` (and its `__all__`).
3. Add the signature to `python/lumapy/_lumapy.pyi`; `stubtest` fails until it matches.
4. Test it against NumPy (or a host reference) in `tests/`.
5. Note it in `CHANGELOG.md`.

## Releasing

1. Bump `version` in `Cargo.toml` (the Python package takes its version from it) and move the changelog entries
   under the new version.
2. Publish a GitHub release tagged `vX.Y.Z`. `release.yml` checks the tag against `Cargo.toml`, builds the wheels
   and the sdist, and publishes them to PyPI with trusted publishing (configure the `release` environment and the
   trusted publisher on PyPI once). The PyPI project is `pyluma`; the import name stays `lumapy`.
