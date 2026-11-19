# LumaPy

Python bindings for [Luma](https://github.com/flipnoteman/Luma), the async GPU compute library for Rust. Write code
with ordinary arithmetic operations and methods, and Luma handles execution with whatever hardware you have (Vulkan,
Metal or DX12, through [wgpu](https://wgpu.rs)).

> **Mirrors Luma 0.4.1** plus the unreleased changes up to
> [`5a592c4`](https://github.com/flipnoteman/Luma/commit/5a592c4c53137ff93e4d0437818d47a24c1f3fbc) (Luma `main`).
> A build reports the Luma it was compiled against: `lumapy.__luma_version__`, `lumapy.__luma_commit__`, or
> `lumapy.show_versions()`.

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
| `Context`, `ContextOptions`, CPU fallback, arrays larger than a buffer or device memory | ✅ `lp.Context(device=, memory_budget=, ...)`; `Context::cost_model` ❌ |

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
- CPU fallback through Mesa llvmpipe when no GPU is present
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

You need Python 3.10 or later and a GPU with Vulkan, Metal or DX12 drivers.

Add LumaPy to a project:

```sh
pip install lumapy        # once a release is on PyPI
```

Or build it from source (needs a Rust toolchain, Luma's `rust-version`, currently 1.99):

```sh
git clone https://github.com/flipnoteman/LumaPy
cd LumaPy
pip install maturin
maturin develop --release
pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup.

### No GPU

Without a GPU, Luma runs through Mesa's llvmpipe, a software Vulkan driver. It is not built into Luma: the driver has
to be installed on the machine (`mesa-vulkan-drivers` on most Linux distributions), otherwise opening a context raises
`lumapy.LumaError`. It is picked automatically when no GPU opens. `LUMA_DEVICE=cpu`, `lp.Context("cpu")` or
`lp.Context.cpu()` force it, and `LUMA_DEVICE=gpu` (or `lp.Context("gpu")`) refuses the fallback and raises instead.

### Large arrays

An array bigger than one device buffer is split over several buffers automatically, and the built-in operations run
over them in pieces. Array data beyond the device memory budget (80% of device memory on Vulkan GPUs) moves to host
memory and back on demand, with a one-time warning (`ctx.spilled_bytes` reports how much). Custom kernels need arrays
that fit one buffer.

### Environment variables

Luma's environment variables apply unchanged. Options passed to `lp.Context(...)` (`device`, `memory_budget`,
`chunk_bytes`, `calibrate`, `check_finite`, `fast_math`, `df_fma`) override them for that context.

| Variable | Effect |
| --- | --- |
| `LUMA_DEVICE` | `gpu` or `cpu`: restrict the device kind (default: GPU, CPU as fallback) |
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
