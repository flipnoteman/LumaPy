# Changelog

All notable changes to LumaPy are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- First Python bindings for Luma 0.4: `lumapy.Array` over every element type (`float32`, `float64`, `df32`,
  `uint32`, `int32`, `bool`) with NumPy-order shapes and axes, NumPy interop (`asarray`, `to_numpy`, `__array__`,
  NumPy operators deferring to the GPU), operators with NumPy-like promotion, elementwise math, reductions with
  `axis`/`keepdims`, scans, sorting, searching, group-by, scatters, filters, views and indexing (including in-place
  assignment through slices and masks), `matmul`, 1-D convolution, tridiagonal solves, `iterate` /
  `iterate_until` / `iterate_all` with Python step functions, `Scalar`, `Context` with its options and profiling,
  and custom WGSL `Kernel`s.
- Type stubs (`_lumapy.pyi`, checked with `stubtest`) and `py.typed`.
- Wheels for Linux, macOS and Windows (abi3, CPython 3.10+) built and published to PyPI on release.
- `lumapy.__luma_version__`, `__luma_commit__` and `__luma_source__` (the Luma the build is compiled against, read
  from `Cargo.lock` at build time) and `lumapy.show_versions()`.
- Python ports of Luma's `basics`, `conv1d` and `custom_kernel` examples, and `examples/datasci.py`.
- `benchmarks/gaussian_nb.py`: Gaussian naive Bayes with LumaPy against scikit-learn's `GaussianNB` and against
  CPU implementations of the same formulation (NumPy on multithreaded BLAS, fused parallel Numba loops), with
  first-call and spread reporting; a test checks that every implementation fits scikit-learn's model.
