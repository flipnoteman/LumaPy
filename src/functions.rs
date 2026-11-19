//! Module-level functions: constructors, `where`, joins, linear algebra, batching.

use luma::{Array, Df32};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::array::PyArr;
use crate::context::PyContext;
use crate::convert::{self, ctx_or_default};
use crate::dtype::{DType, Elem, PyNum};
use crate::error::IntoPy;
use crate::inner::{Inner, extract_shape, luma_axis, luma_shape, norm_axis};
use crate::operand::{Rhs, common, resolve};
use crate::reduce::Mask;
use crate::{with_mask, with_opnd};

fn ctx_arg(ctx: Option<PyRef<'_, PyContext>>) -> PyResult<luma::Context> {
    ctx_or_default(ctx.as_deref().map(|c| &c.inner))
}

fn dtype_arg(dtype: Option<&Bound<'_, PyAny>>, default: DType) -> PyResult<DType> {
    dtype.filter(|d| !d.is_none()).map(DType::from_py).transpose().map(|d| d.unwrap_or(default))
}

/// An array from a NumPy array, nested lists or a scalar. `dtype=None` keeps booleans and
/// 32-bit integers, makes other integers 32-bit and **floats `float32`** (pass
/// `dtype="float64"` or `"df32"` for more precision). Data already in a `lumapy.Array` is not
/// copied.
#[pyfunction]
#[pyo3(signature = (obj, dtype = None, ctx = None))]
pub fn asarray(
    obj: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
    ctx: Option<PyRef<'_, PyContext>>,
) -> PyResult<PyArr> {
    let dtype = dtype.filter(|d| !d.is_none()).map(DType::from_py).transpose()?;
    convert::from_object(obj, dtype, ctx.as_deref().map(|c| &c.inner))
}

/// An array of zeros.
#[pyfunction]
#[pyo3(signature = (shape, dtype = None, ctx = None))]
pub fn zeros(
    py: Python<'_>,
    shape: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
    ctx: Option<PyRef<'_, PyContext>>,
) -> PyResult<PyArr> {
    let shape = extract_shape(shape)?;
    let ls = luma_shape(&shape)?;
    let ctx = ctx_arg(ctx)?;
    let dtype = dtype_arg(dtype, DType::F32)?;
    let inner = match dtype {
        DType::F32 => py.detach(|| Array::<f32>::zeros_on(&ctx, ls)).py()?.into(),
        DType::F64 => Array::<f64>::zeros_on(&ctx, ls).py()?.into(),
        DType::Df32 => Array::<Df32>::zeros_on(&ctx, ls).py()?.into(),
        DType::U32 => Array::<u32>::zeros_on(&ctx, ls).py()?.into(),
        DType::I32 => Array::<i32>::zeros_on(&ctx, ls).py()?.into(),
        DType::Bool => Array::<bool>::zeros_on(&ctx, ls).py()?.into(),
    };
    Ok(PyArr::new(inner, shape.len()))
}

/// An array filled with `fill_value` (dtype from the value when not given).
#[pyfunction]
#[pyo3(signature = (shape, fill_value, dtype = None, ctx = None))]
pub fn full(
    shape: &Bound<'_, PyAny>,
    fill_value: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
    ctx: Option<PyRef<'_, PyContext>>,
) -> PyResult<PyArr> {
    let shape = extract_shape(shape)?;
    let n = PyNum::extract(fill_value).ok_or_else(|| PyTypeError::new_err("fill_value is a number"))?;
    let dtype = dtype_arg(dtype, n.natural_dtype())?;
    let ctx = ctx_arg(ctx)?;
    Ok(PyArr::new(convert::full(&ctx, luma_shape(&shape)?, dtype, n)?, shape.len()))
}

/// Each element's index along `axis` for an array of `shape` (Luma's `Array::range`).
#[pyfunction]
#[pyo3(name = "range", signature = (shape, axis = 0, dtype = None, ctx = None))]
pub fn range_(
    shape: &Bound<'_, PyAny>,
    axis: isize,
    dtype: Option<&Bound<'_, PyAny>>,
    ctx: Option<PyRef<'_, PyContext>>,
) -> PyResult<PyArr> {
    let shape = extract_shape(shape)?;
    let ls = luma_shape(&shape)?;
    let la = luma_axis(axis, shape.len().max(1))?;
    let ctx = ctx_arg(ctx)?;
    let inner: Inner = match dtype_arg(dtype, DType::F32)? {
        DType::F32 => Array::<f32>::range_on(&ctx, ls, la).py()?.into(),
        DType::F64 => Array::<f64>::range_on(&ctx, ls, la).py()?.into(),
        DType::Df32 => Array::<Df32>::range_on(&ctx, ls, la).py()?.into(),
        DType::U32 => Array::<u32>::range_on(&ctx, ls, la).py()?.into(),
        DType::I32 => Array::<i32>::range_on(&ctx, ls, la).py()?.into(),
        DType::Bool => return Err(PyTypeError::new_err("range of bool")),
    };
    Ok(PyArr::new(inner, shape.len()))
}

/// `num` evenly spaced float32 values from `start` to `stop` inclusive.
#[pyfunction]
#[pyo3(signature = (start, stop, num = 50, ctx = None))]
pub fn linspace(start: f32, stop: f32, num: usize, ctx: Option<PyRef<'_, PyContext>>) -> PyResult<PyArr> {
    let ctx = ctx_arg(ctx)?;
    Ok(PyArr::new(Array::linspace_on(&ctx, start, stop, num).py()?.into(), 1))
}

/// Uniform float32 values in `[0, 1)` from a counter-based generator: the same `seed` gives
/// the same values, computed inside the kernels that read them.
#[pyfunction]
#[pyo3(signature = (shape, seed = 0, ctx = None))]
pub fn uniform(shape: &Bound<'_, PyAny>, seed: u64, ctx: Option<PyRef<'_, PyContext>>) -> PyResult<PyArr> {
    let shape = extract_shape(shape)?;
    let ctx = ctx_arg(ctx)?;
    Ok(PyArr::new(Array::uniform_on(&ctx, luma_shape(&shape)?, seed).py()?.into(), shape.len()))
}

/// Standard normal float32 values from a counter-based generator.
#[pyfunction]
#[pyo3(signature = (shape, seed = 0, ctx = None))]
pub fn normal(shape: &Bound<'_, PyAny>, seed: u64, ctx: Option<PyRef<'_, PyContext>>) -> PyResult<PyArr> {
    let shape = extract_shape(shape)?;
    let ctx = ctx_arg(ctx)?;
    Ok(PyArr::new(Array::normal_on(&ctx, luma_shape(&shape)?, seed).py()?.into(), shape.len()))
}

/// Random uint32 bits from a counter-based generator.
#[pyfunction]
#[pyo3(signature = (shape, seed = 0, ctx = None))]
pub fn random_bits(shape: &Bound<'_, PyAny>, seed: u64, ctx: Option<PyRef<'_, PyContext>>) -> PyResult<PyArr> {
    let shape = extract_shape(shape)?;
    let ctx = ctx_arg(ctx)?;
    Ok(PyArr::new(Array::random_bits_on(&ctx, luma_shape(&shape)?, seed).py()?.into(), shape.len()))
}

/// `where(cond, a, b)` for operands already promoted to `dtype`.
pub fn select_inner(mask: &Mask, a: Rhs, b: Rhs, dtype: DType) -> PyResult<Inner> {
    let ctx = match mask {
        Mask::Bool(m) => m.context().clone(),
        Mask::U32(m) => m.context().clone(),
    };
    macro_rules! go {
        ($t:ty, $mode:ident) => {{
            let oa = resolve::<$t>(a, "where")?;
            let ob = resolve::<$t>(b, "where")?;
            with_mask!(mask, m => with_opnd!($mode, oa, &ctx, x => with_opnd!($mode, ob, &ctx, y => Inner::from(luma::select(m, x, y)))))
        }};
    }
    Ok(match dtype {
        DType::F32 => go!(f32, full),
        DType::U32 => go!(u32, full),
        DType::I32 => go!(i32, full),
        DType::Df32 => go!(Df32, val),
        DType::Bool => go!(bool, val),
        DType::F64 => go!(f64, arr),
    })
}

fn operand(obj: &Bound<'_, PyAny>, ctx: &luma::Context) -> PyResult<Rhs> {
    if let Ok(a) = obj.cast::<PyArr>() {
        let a = a.try_borrow()?;
        return Ok(Rhs::Arr(a.inner.clone(), a.ndim));
    }
    if let Ok(s) = obj.cast::<crate::scalar::PyScalar>() {
        return Ok(Rhs::Scalar(s.get().inner.clone()));
    }
    if let Some(n) = PyNum::extract(obj) {
        return Ok(Rhs::Num(n));
    }
    let a = convert::from_object(obj, None, Some(ctx))?;
    Ok(Rhs::Arr(a.inner, a.ndim))
}

fn rhs_dtype(r: &Rhs) -> Option<DType> {
    match r {
        Rhs::Arr(i, _) => Some(i.dtype()),
        Rhs::Scalar(s) => Some(s.dtype()),
        Rhs::Num(_) => None,
    }
}

/// `a` where `cond` is true, else `b`; `a` and `b` are arrays or scalars and broadcast
/// (Luma's `select`, fused into the kernels that read it).
#[pyfunction]
#[pyo3(name = "where")]
pub fn where_(cond: &Bound<'_, PyAny>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<PyArr> {
    let cond_arr = if let Ok(c) = cond.cast::<PyArr>() {
        let c = c.try_borrow()?;
        PyArr::new(c.inner.clone(), c.ndim)
    } else {
        convert::from_object(cond, Some(DType::Bool), None)?
    };
    let ctx = cond_arr.inner.context();
    let (ra, rb) = (operand(a, &ctx)?, operand(b, &ctx)?);
    let dtype = match (rhs_dtype(&ra), rhs_dtype(&rb)) {
        (Some(x), Some(y)) => common(x, y)?,
        (Some(x), None) | (None, Some(x)) => {
            let other = if rhs_dtype(&ra).is_none() { &ra } else { &rb };
            match other {
                Rhs::Num(PyNum::Float(_)) if !x.is_float() => DType::F32,
                Rhs::Num(PyNum::Int(_)) if x == DType::Bool => DType::I32,
                _ => x,
            }
        }
        (None, None) => match (&ra, &rb) {
            (Rhs::Num(PyNum::Float(_)), _) | (_, Rhs::Num(PyNum::Float(_))) => DType::F32,
            (Rhs::Num(PyNum::Int(_)), _) | (_, Rhs::Num(PyNum::Int(_))) => DType::I32,
            _ => DType::Bool,
        },
    };
    let ndim = cond_arr.ndim.max(ra.ndim()).max(rb.ndim());
    let cast = |r: Rhs| -> PyResult<Rhs> {
        Ok(match r {
            Rhs::Arr(i, n) => Rhs::Arr(i.astype(dtype)?, n),
            other => other,
        })
    };
    let mask = Mask::from_arr(cond_arr.inner)?;
    Ok(PyArr::new(select_inner(&mask, cast(ra)?, cast(rb)?, dtype)?, ndim))
}

fn parts(arrays: &Bound<'_, PyAny>) -> PyResult<(Vec<Inner>, usize, DType)> {
    let mut list = Vec::new();
    let mut ndim = 0;
    let mut dtype: Option<DType> = None;
    for item in arrays.try_iter()? {
        let item = item?;
        let arr = if let Ok(a) = item.cast::<PyArr>() {
            let a = a.try_borrow()?;
            PyArr::new(a.inner.clone(), a.ndim)
        } else {
            convert::from_object(&item, None, None)?
        };
        ndim = ndim.max(arr.ndim);
        dtype = Some(match dtype {
            None => arr.inner.dtype(),
            Some(d) => common(d, arr.inner.dtype())?,
        });
        list.push(arr.inner);
    }
    let dtype = dtype.ok_or_else(|| PyValueError::new_err("need at least one array"))?;
    let list = list.into_iter().map(|i| i.astype(dtype)).collect::<PyResult<Vec<_>>>()?;
    Ok((list, ndim, dtype))
}

macro_rules! typed_parts {
    ($list:expr, $dtype:expr, $parts:ident => $e:expr) => {
        match $dtype {
            DType::F32 => {
                let $parts: Vec<Array<f32>> = $list.into_iter().map(|i| f32::take(i).unwrap()).collect();
                Inner::from($e)
            }
            DType::F64 => {
                let $parts: Vec<Array<f64>> = $list.into_iter().map(|i| f64::take(i).unwrap()).collect();
                Inner::from($e)
            }
            DType::Df32 => {
                let $parts: Vec<Array<Df32>> = $list.into_iter().map(|i| Df32::take(i).unwrap()).collect();
                Inner::from($e)
            }
            DType::U32 => {
                let $parts: Vec<Array<u32>> = $list.into_iter().map(|i| u32::take(i).unwrap()).collect();
                Inner::from($e)
            }
            DType::I32 => {
                let $parts: Vec<Array<i32>> = $list.into_iter().map(|i| i32::take(i).unwrap()).collect();
                Inner::from($e)
            }
            DType::Bool => {
                let $parts: Vec<Array<bool>> = $list.into_iter().map(|i| bool::take(i).unwrap()).collect();
                Inner::from($e)
            }
        }
    };
}

/// Joins arrays along an existing axis. Lazy.
#[pyfunction]
#[pyo3(signature = (arrays, axis = 0))]
pub fn concatenate(arrays: &Bound<'_, PyAny>, axis: isize) -> PyResult<PyArr> {
    let (list, ndim, dtype) = parts(arrays)?;
    let la = luma_axis(axis, ndim.max(1))?;
    Ok(PyArr::new(typed_parts!(list, dtype, p => luma::concat_slice(la, &p)), ndim))
}

/// Joins equal-shaped arrays along a new axis (at most three dimensions each).
#[pyfunction]
#[pyo3(signature = (arrays, axis = 0))]
pub fn stack(arrays: &Bound<'_, PyAny>, axis: isize) -> PyResult<PyArr> {
    let (list, ndim, dtype) = parts(arrays)?;
    let a = norm_axis(axis, ndim + 1)?;
    let la = ndim - a;
    Ok(PyArr::new(typed_parts!(list, dtype, p => luma::stack_slice(la, &p)), ndim + 1))
}

/// Starts computing all `arrays` in one submission, without waiting. Raises if one is
/// poisoned.
#[pyfunction]
pub fn eval_all(py: Python<'_>, arrays: Vec<PyRef<'_, PyArr>>) -> PyResult<()> {
    let inners: Vec<Inner> = arrays.iter().map(|a| a.inner.clone()).collect();
    py.detach(|| -> luma::Result<()> {
        macro_rules! group {
            ($($v:ident),*) => {$(
                let typed: Vec<_> = inners.iter().filter_map(|i| match i { Inner::$v(a) => Some(a), _ => None }).collect();
                if !typed.is_empty() {
                    luma::eval_all(&typed)?;
                }
            )*};
        }
        group!(F32, F64, Df32, U32, I32, Bool);
        Ok(())
    })
    .py()
}

/// The matrix product `a @ b`.
#[pyfunction]
pub fn matmul(a: PyRef<'_, PyArr>, b: PyRef<'_, PyArr>) -> PyResult<PyArr> {
    crate::ops::matmul(&a, &b)
}

/// The number of occurrences of each value `< bins` in the uint32 `indices`.
#[pyfunction]
pub fn bincount(indices: &Bound<'_, PyAny>, bins: usize) -> PyResult<PyArr> {
    let arr = if let Ok(a) = indices.cast::<PyArr>() {
        let a = a.try_borrow()?;
        a.inner.astype(DType::U32)?
    } else {
        convert::from_object(indices, Some(DType::U32), None)?.inner
    };
    let Inner::U32(idx) = arr else { unreachable!() };
    Ok(PyArr::new(luma::bincount(&idx, bins).into(), 1))
}

fn f32_arr(obj: &Bound<'_, PyAny>) -> PyResult<(Array<f32>, usize)> {
    let arr = if let Ok(a) = obj.cast::<PyArr>() {
        let a = a.try_borrow()?;
        PyArr::new(a.inner.astype(DType::F32)?, a.ndim)
    } else {
        convert::from_object(obj, Some(DType::F32), None)?
    };
    match arr.inner {
        Inner::F32(a) => Ok((a, arr.ndim)),
        _ => unreachable!(),
    }
}

fn u32_arr(obj: &Bound<'_, PyAny>) -> PyResult<Array<u32>> {
    let arr = if let Ok(a) = obj.cast::<PyArr>() {
        a.try_borrow()?.inner.astype(DType::U32)?
    } else {
        convert::from_object(obj, Some(DType::U32), None)?.inner
    };
    match arr {
        Inner::U32(a) => Ok(a),
        _ => unreachable!(),
    }
}

/// Solves `lo[s] x[s-1] + diag[s] x[s] + up[s] x[s+1] = rhs[s]` along `axis` for every line of
/// the four equal-shaped float32 arrays. `start` (uint32, one per line) begins line `l`'s
/// system at cell `start[l]`, the cells before it getting 0.
#[pyfunction]
#[pyo3(signature = (lo, diag, up, rhs, axis = -1, start = None), text_signature = "(lo, diag, up, rhs, axis=-1, start=None)")]
pub fn solve_tridiagonal(
    lo: &Bound<'_, PyAny>,
    diag: &Bound<'_, PyAny>,
    up: &Bound<'_, PyAny>,
    rhs: &Bound<'_, PyAny>,
    axis: isize,
    start: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyArr> {
    let (lo, _) = f32_arr(lo)?;
    let (diag, _) = f32_arr(diag)?;
    let (up, _) = f32_arr(up)?;
    let (rhs, ndim) = f32_arr(rhs)?;
    let la = luma_axis(axis, ndim)?;
    let x = match start {
        Some(s) => luma::solve_tridiagonal_from(la, &lo, &diag, &up, &rhs, &u32_arr(s)?),
        None => luma::solve_tridiagonal(la, &lo, &diag, &up, &rhs),
    };
    Ok(PyArr::new(x.into(), ndim))
}

/// Block-tridiagonal solve (`block` x `block` blocks, `1 <= block <= 8`) along `axis`. `rhs`
/// has shape `(..., block)` and `lo`, `diag`, `up` `(..., block * block)` (entry `(row, col)`
/// of a block at `row * block + col`); `axis` is one of the grid axes, not the last.
#[pyfunction]
#[pyo3(signature = (block, lo, diag, up, rhs, axis = -2, start = None), text_signature = "(block, lo, diag, up, rhs, axis=-2, start=None)")]
#[allow(clippy::too_many_arguments)]
pub fn solve_block_tridiagonal(
    block: usize,
    lo: &Bound<'_, PyAny>,
    diag: &Bound<'_, PyAny>,
    up: &Bound<'_, PyAny>,
    rhs: &Bound<'_, PyAny>,
    axis: isize,
    start: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyArr> {
    let (lo, _) = f32_arr(lo)?;
    let (diag, _) = f32_arr(diag)?;
    let (up, _) = f32_arr(up)?;
    let (rhs, ndim) = f32_arr(rhs)?;
    let la = luma_axis(axis, ndim)?;
    let x = match start {
        Some(s) => luma::solve_block_tridiagonal_from(block, la, &lo, &diag, &up, &rhs, &u32_arr(s)?),
        None => luma::solve_block_tridiagonal(block, la, &lo, &diag, &up, &rhs),
    };
    Ok(PyArr::new(x.into(), ndim))
}

/// The process-wide default `Context`, opened on first use.
#[pyfunction]
pub fn default_context(py: Python<'_>) -> PyResult<PyContext> {
    Ok(PyContext { inner: py.detach(luma::default_context).py()? })
}

/// Hands the default context's recorded work to the GPU without waiting.
#[pyfunction]
pub fn flush(py: Python<'_>) -> PyResult<()> {
    py.detach(luma::flush).py()
}

/// Calls `fn()` with at most `max_nodes` operations fused into one kernel (1: every operation
/// is a kernel of its own), and returns its result.
#[pyfunction]
pub fn fusion_limit(py: Python<'_>, max_nodes: usize, f: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let _ = py;
    luma::fusion_limit(max_nodes, || f.call0().map(Bound::unbind))
}
