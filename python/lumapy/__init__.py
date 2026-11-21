"""Lazy GPU arrays for Python, on Luma and wgpu (Vulkan, Metal, DX12).

``lumapy.Array`` lives on the GPU. Arithmetic is recorded, fused into kernels and batched;
nothing waits for the GPU until the host needs values::

    import lumapy as lp

    x = lp.asarray([1.0, 2.0, 3.0, 4.0])
    y = lp.asarray([10.0, 20.0, 30.0, 40.0])
    z = lp.sqrt(x * 2 + y)          # recorded, not yet run
    print(z.to_numpy())             # runs the batch and waits
    print((x * y).sum().item())     # 300.0

Shapes and axes follow NumPy (the last axis varies fastest). Floats default to ``float32``;
pass ``dtype="float64"`` (devices with 64-bit floats) or ``dtype="df32"`` (double-float, about
48 bits, every device) for more precision.

A failed operation yields a *poisoned* array instead of raising; the ``LumaError`` comes out
where a result is needed (``to_numpy``, ``item``, ``eval``, ``check``).
"""

from __future__ import annotations

import builtins as _builtins
import math as _math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ._lumapy import (
    Array,
    Context,
    Kernel,
    KernelTiming,
    KernelView,
    LumaError,
    Profile,
    Scalar,
    Uniform,
    __luma_commit__,
    __luma_source__,
    __luma_version__,
    __version__,
    asarray,
    bincount,
    concatenate,
    default_context,
    eval_all,
    flush,
    full,
    fusion_limit,
    iterate_all,
    linspace,
    matmul,
    normal,
    random_bits,
    range,
    solve_block_tridiagonal,
    solve_tridiagonal,
    stack,
    uniform,
    view,
    where,
    zeros,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from ._lumapy import _ArrayLike as ArrayLike
    from ._lumapy import _Axis as Axis
    from ._lumapy import _DTypeLike as DTypeLike
    from ._lumapy import _Operand as Operand
    from ._lumapy import _PadMode as PadMode
    from ._lumapy import _ShapeLike as ShapeLike

__all__ = [
    # classes and errors
    "Array",
    "Context",
    "Kernel",
    "KernelTiming",
    "KernelView",
    "LumaError",
    "Profile",
    "Scalar",
    "Uniform",
    "__luma_commit__",
    "__luma_source__",
    "__luma_version__",
    "__version__",
    "show_versions",
    # dtypes
    "bool_",
    "df32",
    "dtypes",
    "float32",
    "float64",
    "int32",
    "uint32",
    # construction and host transfer
    "arange",
    "array",
    "asarray",
    "full",
    "full_like",
    "linspace",
    "normal",
    "ones",
    "ones_like",
    "random_bits",
    "range",
    "to_numpy",
    "uniform",
    "zeros",
    "zeros_like",
    # elementwise
    "abs",
    "absolute",
    "arctan2",
    "atan2",
    "ceil",
    "clip",
    "copysign",
    "cos",
    "exp",
    "expm1",
    "floor",
    "hypot",
    "isfinite",
    "isinf",
    "isnan",
    "ln",
    "log",
    "log1p",
    "maximum",
    "minimum",
    "power",
    "round",
    "select",
    "sign",
    "sin",
    "sqrt",
    "tan",
    "tanh",
    "trunc",
    "where",
    # reductions and scans
    "all",
    "amax",
    "amin",
    "any",
    "argmax",
    "argmin",
    "count_nonzero",
    "cumsum",
    "iterate_all",
    "max",
    "mean",
    "median",
    "min",
    "nanmax",
    "nanmean",
    "nanmin",
    "nansum",
    "quantile",
    "std",
    "sum",
    "var",
    # sorting, searching, counting
    "argsort",
    "bincount",
    "digitize",
    "histogram",
    "isin",
    "searchsorted",
    "sort",
    "unique",
    # shapes, views, joins
    "broadcast_to",
    "concat",
    "concatenate",
    "expand_dims",
    "flip",
    "pad",
    "ravel",
    "reshape",
    "roll",
    "squeeze",
    "stack",
    "swapaxes",
    "take",
    "transpose",
    # linear algebra
    "dot",
    "matmul",
    "solve_block_tridiagonal",
    "solve_tridiagonal",
    # runtime and kernels
    "default_context",
    "eval_all",
    "flush",
    "fusion_limit",
    "view",
]

# -- versions ------------------------------------------------------------------------------------


def show_versions() -> None:
    """Prints the versions of lumapy, the Luma it is built on, Python and NumPy, and the default
    device (for bug reports)."""
    import platform

    import numpy

    luma = __luma_version__
    if __luma_source__ == "git":
        luma += f" (git {__luma_commit__[:12]})"
    elif __luma_source__ == "local":
        luma += " (local checkout)"
    lines = [
        f"lumapy  {__version__}",
        f"luma    {luma}",
        f"python  {platform.python_version()} ({platform.platform()})",
        f"numpy   {numpy.__version__}",
    ]
    try:
        ctx = default_context()
        lines.append(f"device  {ctx.device_name} ({ctx.backend} backend), tuning profile {ctx.tuning_profile}")
    except LumaError as e:
        lines.append(f"device  none ({e})")
    print("\n".join(lines))


# -- dtypes ---------------------------------------------------------------------------------------

float32 = "float32"
float64 = "float64"
df32 = "df32"
uint32 = "uint32"
int32 = "int32"
bool_ = "bool"

#: The dtypes a Luma array can hold.
dtypes = (float32, float64, df32, uint32, int32, bool_)

# -- Luma names for NumPy ones and the other way round --------------------------------------------

array = asarray
select = where
concat = concatenate


def ones(shape: ShapeLike, dtype: DTypeLike | None = None, ctx: Context | None = None) -> Array:
    """An array of ones (``float32`` by default)."""
    return full(shape, 1, dtype=dtype or float32, ctx=ctx)


def zeros_like(a: ArrayLike, dtype: DTypeLike | None = None) -> Array:
    """Zeros with the shape (and dtype, unless given) of ``a``."""
    a = asarray(a)
    return zeros(a.shape, dtype=dtype or a.dtype, ctx=a.context)


def ones_like(a: ArrayLike, dtype: DTypeLike | None = None) -> Array:
    """Ones with the shape (and dtype, unless given) of ``a``."""
    a = asarray(a)
    return full(a.shape, 1, dtype=dtype or a.dtype, ctx=a.context)


def full_like(a: ArrayLike, fill_value: float, dtype: DTypeLike | None = None) -> Array:
    """``fill_value`` with the shape (and dtype, unless given) of ``a``."""
    a = asarray(a)
    return full(a.shape, fill_value, dtype=dtype or a.dtype, ctx=a.context)


def arange(
    start: float,
    stop: float | None = None,
    step: float = 1,
    dtype: DTypeLike | None = None,
    ctx: Context | None = None,
) -> Array:
    """Evenly spaced values in ``[start, stop)``, as ``np.arange`` (computed on the device)."""
    if stop is None:
        start, stop = 0, start
    ints = _builtins.all(isinstance(v, int) for v in (start, stop, step))
    if ints:
        n = _builtins.max(0, -(-(int(stop) - int(start)) // int(step)))
    else:
        n = _builtins.max(0, _math.ceil((stop - start) / step))
    if n == 0:
        raise ValueError("arange selects no elements (Luma has no empty arrays)")
    if dtype is None:
        dtype = int32 if ints else float32
    idx = range(n, 0, dtype=dtype, ctx=ctx)
    if step != 1:
        idx = idx * step
    if start != 0:
        idx = idx + start
    return idx


def to_numpy(a: ArrayLike) -> npt.NDArray[Any]:
    """The values of ``a`` as a NumPy array; waits for the GPU."""
    return asarray(a).to_numpy()


# -- NumPy-style functions over the methods -------------------------------------------------------


def _unary(name: str, method: str | None = None) -> Callable[[ArrayLike], Array]:
    attr = method or name

    def f(x: ArrayLike) -> Array:
        result: Array = getattr(asarray(x), attr)()
        return result

    f.__name__ = f.__qualname__ = name
    f.__doc__ = f"Elementwise ``{attr}`` (see ``Array.{attr}``)."
    return f


sqrt = _unary("sqrt")
exp = _unary("exp")
log = _unary("log")
ln = _unary("ln")
log1p = _unary("log1p")
expm1 = _unary("expm1")
sin = _unary("sin")
cos = _unary("cos")
tan = _unary("tan")
tanh = _unary("tanh")
floor = _unary("floor")
ceil = _unary("ceil")
trunc = _unary("trunc")
round = _unary("round")
abs = _unary("abs")
absolute = _unary("absolute", "abs")
sign = _unary("sign")
isfinite = _unary("isfinite")
isnan = _unary("isnan")
isinf = _unary("isinf")


def _reduction(name: str) -> Callable[..., Array]:
    def f(a: ArrayLike, axis: Axis = None, keepdims: bool = False) -> Array:
        result: Array = getattr(asarray(a), name)(axis=axis, keepdims=keepdims)
        return result

    f.__name__ = f.__qualname__ = name
    f.__doc__ = f"``Array.{name}`` as a function."
    return f


sum = _reduction("sum")
min = _reduction("min")
max = _reduction("max")
amin = min
amax = max
mean = _reduction("mean")
var = _reduction("var")
std = _reduction("std")
any = _reduction("any")
all = _reduction("all")
count_nonzero = _reduction("count_nonzero")
argmax = _reduction("argmax")
argmin = _reduction("argmin")
nansum = _reduction("nansum")
nanmin = _reduction("nanmin")
nanmax = _reduction("nanmax")
nanmean = _reduction("nanmean")
median = _reduction("median")


def quantile(a: ArrayLike, q: float, axis: int | None = None, keepdims: bool = False) -> Array:
    return asarray(a).quantile(q, axis=axis, keepdims=keepdims)


def cumsum(a: ArrayLike, axis: int | None = None) -> Array:
    return asarray(a).cumsum(axis)


def minimum(a: Operand, b: Operand) -> Array:
    """The elementwise minimum; either side may be a scalar."""
    if isinstance(a, Array) or not isinstance(b, Array):
        return asarray(a).minimum(b)
    return b.minimum(a)


def maximum(a: Operand, b: Operand) -> Array:
    """The elementwise maximum; either side may be a scalar."""
    if isinstance(a, Array) or not isinstance(b, Array):
        return asarray(a).maximum(b)
    return b.maximum(a)


def arctan2(y: ArrayLike, x: Operand) -> Array:
    return asarray(y).atan2(x)


atan2 = arctan2


def hypot(a: ArrayLike, b: Operand) -> Array:
    return asarray(a).hypot(b)


def copysign(a: ArrayLike, b: Operand) -> Array:
    return asarray(a).copysign(b)


def power(a: ArrayLike, b: Operand) -> Array:
    return asarray(a) ** b


def clip(a: ArrayLike, a_min: float | None = None, a_max: float | None = None) -> Array:
    return asarray(a).clip(a_min, a_max)


def reshape(a: ArrayLike, shape: Sequence[int]) -> Array:
    return asarray(a).reshape(shape)


def ravel(a: ArrayLike) -> Array:
    return asarray(a).ravel()


def transpose(a: ArrayLike, axes: Sequence[int] | None = None) -> Array:
    arr = asarray(a)
    return arr.transpose() if axes is None else arr.transpose(axes)


def swapaxes(a: ArrayLike, axis1: int, axis2: int) -> Array:
    return asarray(a).swapaxes(axis1, axis2)


def flip(a: ArrayLike, axis: Axis = None) -> Array:
    return asarray(a).flip(axis)


def roll(a: ArrayLike, shift: int | Sequence[int], axis: Axis = None) -> Array:
    return asarray(a).roll(shift, axis)


def pad(a: ArrayLike, pad_width: int | Sequence[int] | Sequence[tuple[int, int]], mode: PadMode = "constant") -> Array:
    return asarray(a).pad(pad_width, mode)


def broadcast_to(a: ArrayLike, shape: ShapeLike) -> Array:
    return asarray(a).broadcast_to(shape)


def expand_dims(a: ArrayLike, axis: int) -> Array:
    return asarray(a).expand_dims(axis)


def squeeze(a: ArrayLike, axis: Axis = None) -> Array:
    return asarray(a).squeeze(axis)


def sort(a: ArrayLike, axis: int | None = -1) -> Array:
    return asarray(a).sort(axis)


def argsort(a: ArrayLike, axis: int | None = -1) -> Array:
    return asarray(a).argsort(axis)


def unique(a: ArrayLike) -> Array:
    return asarray(a).unique()


def searchsorted(a: ArrayLike, v: ArrayLike, side: str = "left") -> Array:
    return asarray(a).searchsorted(v, side)  # type: ignore[arg-type]


def digitize(x: ArrayLike, bins: ArrayLike, right: bool = False) -> Array:
    return asarray(x).digitize(bins, right)


def isin(element: ArrayLike, test_elements: ArrayLike) -> Array:
    return asarray(element).isin(test_elements)


def take(a: ArrayLike, indices: ArrayLike) -> Array:
    return asarray(a).take(indices)


def dot(a: ArrayLike, b: ArrayLike) -> Array:
    """The product of 1-D or 2-D arrays (``matmul``)."""
    return matmul(asarray(a), asarray(b))


def histogram(a: ArrayLike, bins: int, lo: float, hi: float) -> Array:
    """The counts of ``bins`` equal bins over ``[lo, hi)`` (uint32); see ``Array.histogram``."""
    return asarray(a).histogram(bins, lo, hi)
