# LumaPy

Python bindings for [Luma](https://github.com/flipnoteman/Luma), the async GPU compute library for Rust. Write code
with ordinary arithmetic operations and methods, and Luma handles execution with whatever hardware you have (CUDA,
Metal, or Vulkan/Metal/DX12 through [wgpu](https://wgpu.rs), with a native CPU backend when there is no GPU).

> **Mirrors Luma 0.5.1** ([crates.io](https://crates.io/crates/luma/0.5.1)). A build reports the Luma it was
> compiled against: `lumapy.__luma_version__`, `lumapy.__luma_source__`, or `lumapy.show_versions()`.

```python
import lumapy as lp

# Arrays stored on GPU. The first one opens the default device.
x = lp.asarray([1.0, 2.0, 3.0, 4.0])
y = lp.asarray([10.0, 20.0, 30.0, 40.0])

# Operators and math methods are recorded.
z = (x * 2.0 + y).sqrt()

# Execution deferred until host requires result.
print(z.to_numpy())            # [3.46.. 4.89.. 6. 6.92..]
print((x * y).sum().item())    # 300.0
```

A table of data goes through the same API. Shapes follow NumPy (rows first), so `(4, 3)` below is four rows of
three features:

```python
import numpy as np
import lumapy as lp

x = lp.asarray(np.array([[1, 2, 0], [2, 1, 1], [0, 0, 3], [4, 2, 1]], np.float32))
gram = x.T @ x                     # XᵀX, (3, 3); the transpose is read inside the kernel
variance = x.var(axis=0)           # per feature, in two passes

codes = lp.asarray([0, 1, 0, 1], dtype="uint32")
price = lp.asarray([10.0, 20.0, 30.0, 50.0])
print(price.group_mean(codes, 2).to_numpy())   # [20. 35.]
print(price.median().item())                   # 25.0
print(price.sort(descending=True).to_numpy())  # [50. 30. 20. 10.]
print(gram.shape, variance.size)               # (3, 3) 3
```

## Feature parity with Luma

✅ bound  ·  🟡 partly bound  ·  ❌ not bound yet

| Luma feature | LumaPy |
| --- | --- |
| Arrays of `f32`, `u32`, `i32`, `bool`, `Df32`, `f64`, up to 4 dimensions; casts | ✅ `float32`, `uint32`, `int32`, `bool`, `df32`, `float64`; `astype` |
| `vec2` / `vec4` element types, `fields!` records | ❌ |
| Operators, math functions, comparisons, bitwise ops and shifts, `select`, `clamp`, `shift` | ✅ Python operators, `lp.where`, `clip`, `roll` |
| Views: `slice` / `ViewMut`, `pad`, `flip`, `transpose`, `permute_axes`, `take`, `put`, `concat`, `stack`, `broadcast_to` | ✅ indexing and `x[...] = v`, NumPy-named functions |
| Reductions, axis reductions, `reduce_axis`, `var_axis` / `std_axis`, `scan` / `cumsum` | ✅ with NumPy's `axis` / `keepdims` |
| `solve_tridiagonal`, `solve_block_tridiagonal` (and `_from`) | ✅ |
| `solve_tridiagonal_parts`, `solve_block_tridiagonal_parts` | ❌ |
| Data science: `matmul`, scatters, `bincount`, `histogram`, sorts, group-by, `factorize` / `unique`, `quantile` / `median`, `searchsorted` / `digitize` / `isin`, NaN-aware reductions | ✅ |
| Counter-based random numbers, `range`, `linspace` | ✅ |
| Lazy evaluation and kernel fusion, in-place compound assignment, batched submission | ✅ (Luma's engine, unchanged) |
| Non-blocking `eval` / `eval_all`, `to_vec` / `read_into` | ✅ `eval`, `eval_all`, `to_numpy` / `np.asarray`, `read_into` |
| `iterate`, `iterate_all`, `iterate_until` | ✅ with Python step functions |
| `iterate_until_all`, `compile_until_all`, `Until` / `History` / `Stat` / `Cond` / `Run` | ❌ |
| Custom WGSL kernels (`Kernel`, `view`, `uniform`) | 🟡 uniforms are single `f32` / `u32` / `i32` values, not structs |
| `Scalar` (values changed without recompiling) | ✅ |
| Subgroups, `f16`, cooperative matrices, per-vendor kernel tuning | ✅ (automatic inside Luma) |
| Errors as values (poisoned arrays) | ✅ raised as `lumapy.LumaError` where a result is needed |
| Per-kernel GPU timing (`Context::profile`) | ✅ `ctx.profile(fn)` |
| `Context`, `ContextOptions` (`backend`, `device`, ...), `Context::{backend, device_name}`, CPU backend, arrays larger than a buffer or device memory | ✅ `lp.Context(device=, backend=, memory_budget=, ...)`, `ctx.backend`, `ctx.device_name`; `Context::cost_model` ❌ |
| `Context::memory_stats`, `Context::trim` | ❌ |

Similar to [ArrayFire](https://arrayfire.com), code is written with the familiarity of the `Array` interface. Execution
of operations is deferred until the host requires the result. Chains of elementwise operations are fused into a single
kernel, and all intermediate work is batched on submission so the GPU doesn't waste time on many small, I/O dependent,
operations.

Luma's **[user guide](https://flipnoteman.github.io/Luma/)** explains the model (lazy evaluation and fusion, views,
loops, precision) with Rust examples; the concepts apply unchanged to LumaPy. The Python API is typed (`py.typed`
and stubs), so editors show every signature.

## Features

- Arrays of `float32`, `uint32`, `int32`, `bool` (the result of comparisons, stored as one bit per element), `df32`
  (double-float, about 48-bit) and `float64` (where supported) with up to 4 dimensions; casts between them with
  `astype`; bitwise operations and shifts on `uint32` / `int32`
- NumPy interop: `lp.asarray(ndarray)`, `x.to_numpy()`, `np.asarray(x)`; NumPy operators on a `lumapy.Array` run on
  the GPU
- Built-in arithmetic operators (with NumPy-like type promotion), math functions, comparisons, `lp.where` (branches
  can be arrays or scalars), `clip`, circular `roll` / `shift`
- Views: indexing with ints, slices, `None`, `...` and masks, assignment through them (`x[1:3, :] = v`,
  `x[x < 0] = 0`), `pad`, `flip`, `transpose` / `T`, `swapaxes`, `take`, `put`, `lp.concatenate`, `lp.stack`,
  `broadcast_to`, `reshape`
- Reductions: `sum`, `min`, `max`, `mean`, `var`, `std`, `norm`, over all axes or any of them (`axis=`, `keepdims=`);
  `reduce` / `reduce_axis` compute several in one pass; `scan` / `cumsum`; `lp.solve_tridiagonal`
- Data-science primitives: `@` / `lp.matmul`; atomic scatters `put_add` / `put_min` / `put_max`, `lp.bincount` and
  `histogram`; radix `sort`, `argsort`, sorts along an axis; `group_sum` / `group_mean` / `group_min` / `group_max` /
  `group_count`, `factorize` / `unique`; `quantile` and `median`; `searchsorted` / `digitize` / `isin`; NaN-aware
  reductions (`nansum`, `nanmean`, `nanmin`, `nanmax`); `round`, `trunc`, `log1p`, `expm1`. Pending operands are
  computed inside their kernels instead of being stored first
- Counter-based random numbers (`lp.uniform`, `lp.normal`, `lp.random_bits`), `lp.range`, `lp.arange` and
  `lp.linspace`, computed inside the kernels that read them instead of stored
- Lazy evaluation: elementwise chains compile into a single kernel; compound assignment (`x += ..`) fuses too and
  overwrites the array's buffer in place
- Batched submission
- Non-blocking `eval` / `lp.eval_all`: the GPU runs while Python records the next steps; reads and `sync` wait
  (with the GIL released)
- Readback into a new NumPy array (`to_numpy`) or into one you already own (`read_into`)
- Multi-step stencils with `iterate`; elementwise loops over several arrays with `lp.iterate_all`, compiled into one
  kernel that keeps the state in registers; `iterate_until` stops on a device-side condition
- Ability to use custom WGSL kernels (`lp.Kernel`)
- Subgroups, `f16` and cooperative matrices (tensor cores) on GPUs that support them
- Kernel tuning per GPU vendor (Apple and NVIDIA measured, generic defaults elsewhere)
- Useful error handling: failed operations raise `lumapy.LumaError` with the operation and its cause
- Per-kernel GPU timing with `ctx.profile(..)`
- Native CUDA and Metal backends next to wgpu, chosen automatically (`lp.Context(backend=...)` or `LUMA_BACKEND` to
  pick one), and a native CPU backend (Cranelift SIMD on all cores) when no GPU is present
- Arrays larger than one device buffer or than device memory

### Conventions

- **NumPy shapes and axes.** `shape`, `axis=`, indexing, broadcasting and `@` mean what they mean in NumPy (the last
  axis varies fastest). Luma lists the fastest axis first; the bindings reverse shapes and axes on the way in, so data
  moves between NumPy and the GPU without a transpose. `x.luma_shape` shows Luma's view. Arrays have no zero-sized
  axes.
- **Lazy.** Nothing waits for the GPU until a value is needed (`to_numpy`, `item`, `tolist`, `repr`, `float(x)`).
- **Errors do not raise where they happen.** A failed operation yields a *poisoned* array; `lumapy.LumaError` is
  raised where a result is needed. `x.error` and `x.check()` inspect it without reading the data.
- **dtypes.** Without `dtype=`, **floats become `float32`** and integers 32-bit, as in other GPU array libraries;
  64-bit integers that do not fit raise. `df32` data travels to and from NumPy as `float64`.
- **Promotion** is NumPy-like within those types: integers and booleans join floats, `bool` joins anything, `/` of
  integers gives `float32`, `//` and `%` round toward negative infinity (Luma's truncating `%` is `rem()`). Mixing
  `uint32` with `int32` (or `float64` with `df32`) raises; convert one with `astype`.
- **Views share data.** Slices, `transpose`, `flip`, `reshape`, `pad` and `roll` are lazy read views. In-place
  operations (`+=`, `x[...] = v`, `put_add`) overwrite the data, which every handle on it sees; `copy()` makes an
  independent array. Expressions recorded before a write keep the values they were recorded against.

## Getting started

You need Python 3.10 or later. A GPU is optional: Luma uses an NVIDIA GPU through its driver (no CUDA toolkit
needed), Apple GPUs through Metal, other GPUs through Vulkan or DX12, and the CPU otherwise.

Add LumaPy to a project:

```sh
pip install pyluma        # once a release is on PyPI; then `import lumapy`
```

Prebuilt wheels are published for Linux (x86_64, aarch64) and Windows (x64). On macOS, `pip install pyluma`
builds from the source distribution, which needs a Rust toolchain.

Or build it from source (needs a Rust toolchain, Luma's `rust-version`, currently 1.99):

```sh
git clone https://github.com/flipnoteman/LumaPy
cd LumaPy
pip install maturin
maturin develop --release
pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup.

### Backends and no GPU

`backend="auto"` (the default) opens the first of CUDA, Metal and wgpu that has a GPU, and otherwise Luma's native
CPU backend, which is built into every wheel (no system driver needed). `lp.Context(backend="wgpu")` or
`LUMA_BACKEND=wgpu` picks one; `ctx.backend` and `ctx.device_name` say which device runs. `LUMA_DEVICE=cpu`,
`lp.Context("cpu")` or `lp.Context.cpu()` force the CPU backend, and `LUMA_DEVICE=gpu` (or `lp.Context("gpu")`)
refuses it and raises instead. Software Vulkan drivers (llvmpipe, SwiftShader, WARP) are never used.

### Large arrays

An array bigger than one device buffer is split over several buffers automatically, and the built-in operations run
over them in pieces. Array data beyond the device memory budget (80% of device memory on GPUs) moves to host
memory and back on demand, with a one-time warning (`ctx.spilled_bytes` reports how much). Custom kernels need arrays
that fit one buffer.

### Environment variables

Luma's environment variables apply unchanged. Options passed to `lp.Context(...)` (`device`, `backend`,
`memory_budget`, `chunk_bytes`, `calibrate`, `check_finite`, `fast_math`, `df_fma`) override them for that context.

| Variable | Effect |
| --- | --- |
| `LUMA_BACKEND` | `cuda`, `metal`, `wgpu`, `cpu` or `auto`: the backend to open (default `auto`: CUDA, then Metal, then wgpu, then the CPU) |
| `LUMA_DEVICE` | `gpu` or `cpu`: restrict the device kind (default: GPU, CPU as fallback) |
| `LUMA_CPU_THREADS` | Threads of the CPU backend (default: the available parallelism) |
| `LUMA_MEMORY_BUDGET` | Bytes of array data kept on the device before spilling to host memory (idle buffers kept for reuse take up to half of it) |
| `LUMA_CHUNK_BYTES` | Largest buffer an array is split into, in bytes (power of two) |
| `LUMA_TUNING` | `apple`, `nvidia`, `generic` or `cpu`: override the kernel tuning profile |
| `LUMA_SUBGROUPS` | `off` keeps the built-in kernels off subgroup operations |
| `LUMA_FUSION_MAX_NODES` | Most operations fused into one kernel (default 16384, at most 16384); an expression nests at most 512 levels |
| `LUMA_FUSION_MAX_LIVE` | Most values a fused kernel keeps live at once before it is split in two (default depends on the GPU profile, `8..=4096`) |
| `LUMA_FUSION_MAX_INLINED` | Most operations of one kernel after nested views of lazy expressions are unfolded into it; a bigger kernel evaluates the most repeated view first (default 1536 on NVIDIA, 1024 on other GPUs, 192 on CPU; `16..=16384`) |
| `LUMA_STAGING` | `off` keeps expressions read through slices and pads at several offsets (fluxes, stencils of an expression) out of workgroup memory: the kernel recomputes them per read instead of computing each once per cell of a workgroup tile (the cost model picks between them) |
| `LUMA_TRIDIAGONAL` | `thomas` or `cyclic`: force the algorithm of `solve_tridiagonal` / `solve_block_tridiagonal` (one invocation per line, or one workgroup per line by cyclic reduction); the default picks cyclic reduction for few lines |
| `LUMA_SPECULATE` | `off` stops readbacks in a loop (`step` then `.max().item()` every iteration) from computing the next steps ahead |
| `LUMA_FAST_MATH` | `1`, `on` or `true`: allow rewrites that change values (`x + 0`, `x * 0`, `x - x`, `x / x`, constant reassociation, `x / c` as `x * (1 / c)`, `powf` with exponent 1, 2 or 0.5) and the cheaper `df32` addition, whose error grows with the cancellation (below 2^-47 of the larger operand); exact rewrites (`x * 1`, `-(-x)`) always run |
| `LUMA_DF_FMA` | `off` makes `df32` products split their factors (Dekker) instead of taking the error of a product from `fma`; `on` trusts that the device's `fma` is fused. Default: a one-time test kernel decides per device (stored with the calibration, see `LUMA_CACHE`); the results are the same either way |
| `LUMA_CHECK_FINITE` | `1`, `on` or `true`: fail any operation whose result holds a NaN or Inf (debug mode, waits for the GPU after every evaluated array) |
| `LUMA_CALIBRATE` | `0`, `off` or `false`: skip measuring the device's throughputs when a context opens (about 150 ms on a GPU, once per machine: a clean result is kept in the cache directory, see `LUMA_CACHE`; a measurement that looks contended by other processes is dropped) and use the tuning profile's fixed cost model |
| `LUMA_CACHE` | `0`, `off` or `false`: do not keep anything between processes. By default luma stores the driver's compiled pipelines (wgpu `PipelineCache`, Vulkan) and the measured cost model per adapter and driver, so a second run skips the multi-second compile of large (`df32`) kernels and the calibration |
| `LUMA_CACHE_DIR` | Directory of that cache (default `$XDG_CACHE_HOME/luma`, else `~/.cache/luma`); files are written atomically, so processes may share it, and a damaged file is ignored |
| `LUMA_EXIT_WAIT` | `0` skips waiting at exit for NVIDIA's driver threads on Linux (saves about 50 ms; risks the driver's exit hang or abort). By default exit waits up to 2 s for the GPU and up to 2 s for busy driver compiler threads and then ends the process with `_exit` |

### Examples

Python ports of Luma's examples are in [`examples/`](examples):

```sh
python examples/basics.py           # expressions, select, reductions
python examples/conv1d.py           # 1D convolution
python examples/custom_kernel.py    # custom WGSL kernels
python examples/datasci.py          # matmul, axis reductions, group-by, median, sort
```

Luma's other examples (benchmarks, Keccak / SHA-3, ML-KEM) exercise the Rust API and stay in the
[Luma repository](https://github.com/flipnoteman/Luma/tree/main/examples).

### Benchmarks

[`benchmarks/gaussian_nb.py`](benchmarks/gaussian_nb.py) compares a Gaussian naive Bayes classifier on the GPU
with CPU implementations of the same model. Every implementation reproduces scikit-learn's `GaussianNB` (same
priors, means, variances, smoothing and log-likelihood; `tests/test_gaussian_nb.py` checks them) and predicts the
same labels as scikit-learn on every test sample. To separate what the GPU contributes from what a better
algorithm contributes, the CPU side includes the same matrix-product formulation the GPU version uses, in NumPy
(on multithreaded BLAS), and as fused, parallel loops compiled with Numba; speedups are quoted against the
fastest CPU implementation.

```sh
pip install --group bench
python benchmarks/gaussian_nb.py            # --quick for a smoke run, --threads N, --help for sizes and options
```

**In short:** the total time to train the classifier and predict with it (lower is better). All four give exactly
the same predictions.

| Problem size | scikit-learn (the standard tool) | Hand-optimized CPU code (all 24 cores) | LumaPy on the GPU, data copied over first | LumaPy on the GPU, data already there | Fastest |
| --- | ---: | ---: | ---: | ---: | --- |
| Small: 10,000 rows × 16 columns | 2.1 ms | **0.29 ms** | 0.47 ms | 0.37 ms | CPU |
| Medium: 100,000 × 32 | 37 ms | 1.7 ms | 2.3 ms | **0.49 ms** | GPU (data already there) |
| Large: 1,000,000 × 32 | 417 ms | 24 ms | 13 ms | **2.2 ms** | GPU |
| Wide: 200,000 × 256 | 423 ms | 17 ms | 21 ms | **3.0 ms** | GPU (data already there) |

Small jobs are faster on the CPU: handing work to the GPU costs more than the whole job. Big jobs are 6–11× faster
on the GPU than well-optimized CPU code, and most so when the data already lives on the GPU. The first run of a new
problem size also spends up to about 0.4 s preparing GPU code (about 0.1 s once Luma's disk cache has it).

The detailed results, fit and predict separately:

On an NVIDIA GeForce RTX 4070 (Luma 0.5.1, CUDA backend) and an AMD Ryzen 9 7900X (12 cores, 24 threads), 10
classes, median of 9 runs after a warm-up; the test set is a quarter of the training set. *From NumPy* includes
uploading the arrays to the GPU; *GPU data* starts from arrays already on it. The fastest CPU implementation was
Numba everywhere.

| train samples × features | scikit-learn fit / predict | Numba (24 threads) fit / predict | LumaPy from NumPy fit / predict | LumaPy GPU data fit / predict |
| --- | ---: | ---: | ---: | ---: |
| 10,000 × 16 | 1.43 ms / 0.70 ms | 0.25 ms / 0.04 ms | 0.40 ms / 0.07 ms (0.6× / 0.5×) | 0.31 ms / 0.06 ms (0.8× / 0.6×) |
| 100,000 × 32 | 18.8 ms / 17.8 ms | 1.08 ms / 0.58 ms | 2.00 ms / 0.32 ms (0.5× / 1.8×) | 0.44 ms / 0.05 ms (2.4× / 11.7×) |
| 1,000,000 × 32 | 216 ms / 201 ms | 19.4 ms / 4.23 ms | 9.82 ms / 2.99 ms (2.0× / 1.4×) | 1.85 ms / 0.33 ms (10.5× / 12.9×) |
| 200,000 × 256 | 156 ms / 267 ms | 10.5 ms / 6.27 ms | 17.2 ms / 4.26 ms (0.6× / 1.5×) | 2.55 ms / 0.43 ms (4.1× / 14.6×) |

Speedups in parentheses are against Numba. What the numbers say:

* Against scikit-learn as shipped LumaPy is 5–620× faster, but most of that is the algorithm: scikit-learn loops
  over the classes in Python, in float64, mostly on one core. The same model as parallel CPU loops (Numba) is
  already 6–48× faster than scikit-learn.
* Against that CPU baseline the GPU wins when the data is already on the GPU (2–15× from 100,000 samples on),
  which is the case in a pipeline that keeps its arrays on the device.
* When the data starts in NumPy, uploading it dominates: below about a million samples the 24-thread CPU is as
  fast or faster. Small problems (10,000 samples) are faster on the CPU either way; a GPU launch costs more than
  the whole computation.
* The first call of a new shape compiles kernels (up to 0.4 s here, about 0.1 s from Luma's disk cache in later
  processes), so a single fit on fresh shapes is slower on the GPU; the benchmark reports it separately.

All implementations work in float32 except scikit-learn and the float64 NumPy rows, which the benchmark also
reports; Numba accumulates its sums in float64.

Luma's Criterion benchmarks of real workloads (Black-Scholes, Mandelbrot, n-body, SGEMM, 2D heat, FIR, k-means, MLP,
Monte Carlo, imaging) compare CPU and GPU execution in Rust; see
[Luma's README](https://github.com/flipnoteman/Luma#benchmarks). The bindings add one Python call per recorded
operation on the host; the GPU work is the same kernels.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in
LumaPy by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.
