//! Reductions, scans, sorting, searching, group-by and scatters on `lumapy.Array`.

use luma::{Array, Df32, Reduction, Side};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::array::PyArr;
use crate::convert;
use crate::dtype::DType;
use crate::error::IntoPy;
use crate::inner::{Inner, luma_axis, norm_axis, py_dims};
use crate::operand::{Rhs, resolve};
use crate::{dispatch, with_opnd};

/// NumPy's `axis` argument normalized to `0..ndim`: `None` (all axes), an int or a tuple.
pub fn extract_axes(axis: Option<&Bound<'_, PyAny>>, ndim: usize) -> PyResult<Option<Vec<usize>>> {
    let Some(axis) = axis.filter(|a| !a.is_none()) else {
        return Ok(None);
    };
    let list: Vec<isize> = match axis.extract::<isize>() {
        Ok(a) => vec![a],
        Err(_) => axis.extract().map_err(|_| PyTypeError::new_err("axis is None, an int or a tuple of ints"))?,
    };
    let mut out = Vec::with_capacity(list.len());
    for a in list {
        let a = norm_axis(a, ndim)?;
        if out.contains(&a) {
            return Err(PyValueError::new_err("repeated axis"));
        }
        out.push(a);
    }
    Ok(Some(out))
}

type Full<'a> = &'a dyn Fn(&Inner) -> PyResult<Inner>;
type Along<'a> = &'a dyn Fn(&Inner, usize) -> PyResult<Inner>;

impl PyArr {
    /// A reduction over all axes (`full`, giving one element) or over the listed NumPy axes
    /// (`along`, called per Luma axis, keeping extent 1), with NumPy's `keepdims`.
    pub fn reduce_with(
        &self,
        axis: Option<&Bound<'_, PyAny>>,
        keepdims: bool,
        several: bool,
        full: Full<'_>,
        along: Along<'_>,
    ) -> PyResult<PyArr> {
        match extract_axes(axis, self.ndim)? {
            None => {
                let r = full(&self.inner)?;
                if keepdims { PyArr::new(r, 0).with_shape(&vec![1; self.ndim]) } else { Ok(PyArr::new(r, 0)) }
            }
            Some(axes) => {
                if !several && axes.len() > 1 {
                    return Err(PyValueError::new_err("this reduction takes a single axis"));
                }
                let mut cur = self.inner.clone();
                for &a in &axes {
                    cur = along(&cur, self.ndim - 1 - a)?;
                }
                if keepdims {
                    return Ok(PyArr::new(cur, self.ndim));
                }
                let kept: Vec<usize> = py_dims(cur.shape(), self.ndim)
                    .into_iter()
                    .enumerate()
                    .filter(|(i, _)| !axes.contains(i))
                    .map(|(_, d)| d)
                    .collect();
                PyArr::new(cur, 0).with_shape(&kept)
            }
        }
    }

    /// The array as one flat axis.
    pub fn flat(&self) -> Inner {
        self.inner.reshape(self.inner.len().into())
    }
}

fn mean_full(inner: &Inner) -> PyResult<Inner> {
    let n = inner.len();
    Ok(match inner {
        Inner::F32(a) => a.mean().into(),
        Inner::F64(a) => (&a.sum() / n as f64).into(),
        Inner::Df32(a) => (&a.sum() / Df32::from(n as f64)).into(),
        other => mean_full(&other.astype(DType::F32)?)?,
    })
}

fn mean_along(inner: &Inner, axis: usize) -> PyResult<Inner> {
    let n = inner.shape().dims()[axis];
    Ok(match inner {
        Inner::F32(a) => a.mean_axis(axis).into(),
        Inner::F64(a) => (&a.sum_axis(axis) / n as f64).into(),
        Inner::Df32(a) => (&a.sum_axis(axis) / Df32::from(n as f64)).into(),
        other => mean_along(&other.astype(DType::F32)?, axis)?,
    })
}

fn f32_of(inner: &Inner, op: &str) -> PyResult<Array<f32>> {
    match inner {
        Inner::F32(a) => Ok(a.clone()),
        Inner::F64(_) | Inner::Df32(_) => {
            Err(PyTypeError::new_err(format!("{op} is float32 only; convert with .astype(\"float32\")")))
        }
        other => match other.astype(DType::F32)? {
            Inner::F32(a) => Ok(a),
            _ => unreachable!(),
        },
    }
}

fn reduction(name: &str) -> PyResult<Reduction> {
    Ok(match name {
        "sum" => Reduction::Sum,
        "min" => Reduction::Min,
        "max" => Reduction::Max,
        "sumsq" | "sum_of_squares" => Reduction::SumOfSquares,
        other => return Err(PyValueError::new_err(format!("unknown reduction {other:?} (sum, min, max, sumsq)"))),
    })
}

/// A mask for Luma's `Condition`: `bool` or `uint32` (non-zero) arrays; others become `bool`.
pub enum Mask {
    Bool(Array<bool>),
    U32(Array<u32>),
}

impl Mask {
    pub fn from_py(obj: &Bound<'_, PyAny>, like: &PyArr) -> PyResult<(Mask, usize)> {
        let arr = if let Ok(a) = obj.cast::<PyArr>() {
            let a = a.try_borrow()?;
            PyArr::new(a.inner.clone(), a.ndim)
        } else {
            convert::from_object(obj, Some(DType::Bool), Some(&like.inner.context()))?
        };
        Ok((Mask::from_arr(arr.inner)?, arr.ndim))
    }

    pub fn from_arr(inner: Inner) -> PyResult<Mask> {
        Ok(match inner {
            Inner::Bool(b) => Mask::Bool(b),
            Inner::U32(u) => Mask::U32(u),
            other => match other.astype(DType::Bool)? {
                Inner::Bool(b) => Mask::Bool(b),
                _ => unreachable!(),
            },
        })
    }
}

/// Runs `$e` with `$m` bound to the mask as a `&impl Condition`.
#[macro_export]
macro_rules! with_mask {
    ($mask:expr, $m:ident => $e:expr) => {
        match $mask {
            $crate::reduce::Mask::Bool(x) => {
                let $m = x;
                $e
            }
            $crate::reduce::Mask::U32(x) => {
                let $m = x;
                $e
            }
        }
    };
}

/// An index array (`uint32`): a `lumapy.Array` is converted, anything else uploaded.
pub fn indices(obj: &Bound<'_, PyAny>, like: &PyArr) -> PyResult<(Array<u32>, usize)> {
    let arr = if let Ok(a) = obj.cast::<PyArr>() {
        let a = a.try_borrow()?;
        PyArr::new(a.inner.astype(DType::U32)?, a.ndim)
    } else {
        convert::from_object(obj, Some(DType::U32), Some(&like.inner.context()))?
    };
    match arr.inner {
        Inner::U32(u) => Ok((u, arr.ndim)),
        _ => unreachable!(),
    }
}

/// A second array of this array's dtype (searches, `isin`).
fn same_dtype(obj: &Bound<'_, PyAny>, like: &PyArr) -> PyResult<PyArr> {
    if let Ok(a) = obj.cast::<PyArr>() {
        let a = a.try_borrow()?;
        return Ok(PyArr::new(a.inner.astype(like.inner.dtype())?, a.ndim));
    }
    convert::from_object(obj, Some(like.inner.dtype()), Some(&like.inner.context()))
}

/// The values operand of a scatter, converted to the target's dtype.
fn values_rhs(obj: &Bound<'_, PyAny>, target: &PyArr) -> PyResult<Rhs> {
    let rhs = Rhs::extract(obj, target, true)?.ok_or_else(|| PyTypeError::new_err("values is an array or a scalar"))?;
    Ok(match rhs {
        Rhs::Arr(inner, ndim) => Rhs::Arr(inner.astype(target.inner.dtype())?, ndim),
        other => other,
    })
}

/// A mutating operation over the listed `Variant: type: mode` triples, `$a` bound to
/// `&mut Array<T>`; returns `PyResult<()>`.
macro_rules! mut_op {
    ($inner:expr, $rhs:expr, $name:expr, [$($v:ident : $t:ty : $mode:ident),+], $a:ident, $b:ident => $e:expr) => {{
        let rhs: Rhs = $rhs;
        match $inner {
            $(
                Inner::$v($a) => {
                    let ctx = $a.context().clone();
                    let _ = &ctx;
                    let o = resolve::<$t>(rhs, $name)?;
                    with_opnd!($mode, o, &ctx, $b => $e).py()
                }
            )+
            #[allow(unreachable_patterns)]
            other => Err(crate::inner::unsupported($name, other.dtype())),
        }
    }};
}

#[pymethods]
impl PyArr {
    /// The sum over `axis` (all axes when `None`). Sums of `bool` are counts (`uint32`).
    #[pyo3(signature = (axis = None, keepdims = false))]
    fn sum(&self, axis: Option<&Bound<'_, PyAny>>, keepdims: bool) -> PyResult<PyArr> {
        self.reduce_with(
            axis,
            keepdims,
            true,
            &|x| dispatch!(x, "sum", [F32, F64, Df32, U32, I32, Bool], a => Inner::from(a.sum())),
            &|x, ax| dispatch!(x, "sum", [F32, F64, Df32, U32, I32, Bool], a => Inner::from(a.sum_axis(ax))),
        )
    }

    /// The minimum over `axis`.
    #[pyo3(signature = (axis = None, keepdims = false))]
    fn min(&self, axis: Option<&Bound<'_, PyAny>>, keepdims: bool) -> PyResult<PyArr> {
        self.reduce_with(
            axis,
            keepdims,
            true,
            &|x| dispatch!(x, "min", [F32, F64, Df32, U32, I32, Bool], a => Inner::from(a.min())),
            &|x, ax| dispatch!(x, "min", [F32, F64, Df32, U32, I32, Bool], a => Inner::from(a.min_axis(ax))),
        )
    }

    /// The maximum over `axis`.
    #[pyo3(signature = (axis = None, keepdims = false))]
    fn max(&self, axis: Option<&Bound<'_, PyAny>>, keepdims: bool) -> PyResult<PyArr> {
        self.reduce_with(
            axis,
            keepdims,
            true,
            &|x| dispatch!(x, "max", [F32, F64, Df32, U32, I32, Bool], a => Inner::from(a.max())),
            &|x, ax| dispatch!(x, "max", [F32, F64, Df32, U32, I32, Bool], a => Inner::from(a.max_axis(ax))),
        )
    }

    /// The mean over `axis` (integers give float32).
    #[pyo3(signature = (axis = None, keepdims = false))]
    fn mean(&self, axis: Option<&Bound<'_, PyAny>>, keepdims: bool) -> PyResult<PyArr> {
        self.reduce_with(axis, keepdims, true, &mean_full, &mean_along)
    }

    /// The population variance (`ddof=0`) over one axis or all of them; float32.
    #[pyo3(signature = (axis = None, keepdims = false))]
    fn var(&self, axis: Option<&Bound<'_, PyAny>>, keepdims: bool) -> PyResult<PyArr> {
        self.reduce_with(
            axis,
            keepdims,
            false,
            &|x| Ok(f32_of(&x.reshape(x.len().into()), "var")?.var_axis(0).into()),
            &|x, ax| Ok(f32_of(x, "var")?.var_axis(ax).into()),
        )
    }

    /// The population standard deviation (`ddof=0`) over one axis or all of them; float32.
    #[pyo3(signature = (axis = None, keepdims = false))]
    fn std(&self, axis: Option<&Bound<'_, PyAny>>, keepdims: bool) -> PyResult<PyArr> {
        self.reduce_with(
            axis,
            keepdims,
            false,
            &|x| Ok(f32_of(&x.reshape(x.len().into()), "std")?.std_axis(0).into()),
            &|x, ax| Ok(f32_of(x, "std")?.std_axis(ax).into()),
        )
    }

    /// The Euclidean norm of all elements; float32.
    fn norm(&self) -> PyResult<PyArr> {
        Ok(PyArr::new(f32_of(&self.inner, "norm")?.norm().into(), 0))
    }

    /// The sum with compensated (Kahan-Babuska) summation; float32.
    fn sum_compensated(&self) -> PyResult<PyArr> {
        Ok(PyArr::new(f32_of(&self.inner, "sum_compensated")?.sum_compensated().into(), 0))
    }

    /// The norm with compensated summation; float32.
    fn norm_compensated(&self) -> PyResult<PyArr> {
        Ok(PyArr::new(f32_of(&self.inner, "norm_compensated")?.norm_compensated().into(), 0))
    }

    /// Whether any element is true (non-zero) over `axis`.
    #[pyo3(signature = (axis = None, keepdims = false))]
    fn any(&self, axis: Option<&Bound<'_, PyAny>>, keepdims: bool) -> PyResult<PyArr> {
        let b = self.same(self.inner.astype(DType::Bool)?);
        b.reduce_with(
            axis,
            keepdims,
            true,
            &|x| dispatch!(x, "any", [Bool], a => Inner::from(a.any())),
            &|x, ax| dispatch!(x, "any", [Bool], a => Inner::from(a.max_axis(ax))),
        )
    }

    /// Whether all elements are true (non-zero) over `axis`.
    #[pyo3(signature = (axis = None, keepdims = false))]
    fn all(&self, axis: Option<&Bound<'_, PyAny>>, keepdims: bool) -> PyResult<PyArr> {
        let b = self.same(self.inner.astype(DType::Bool)?);
        b.reduce_with(
            axis,
            keepdims,
            true,
            &|x| dispatch!(x, "all", [Bool], a => Inner::from(a.all())),
            &|x, ax| dispatch!(x, "all", [Bool], a => Inner::from(a.min_axis(ax))),
        )
    }

    /// The number of true (non-zero) elements over `axis`, as uint32.
    #[pyo3(signature = (axis = None, keepdims = false))]
    fn count_nonzero(&self, axis: Option<&Bound<'_, PyAny>>, keepdims: bool) -> PyResult<PyArr> {
        let b = self.same(self.inner.astype(DType::Bool)?);
        b.sum(axis, keepdims)
    }

    /// Several reductions in one pass over the data (up to four of `"sum"`, `"min"`, `"max"`,
    /// `"sumsq"`), as a 1-D array with one element per reduction.
    fn reduce(&self, ops: Vec<String>) -> PyResult<PyArr> {
        let ops: Vec<Reduction> = ops.iter().map(|o| reduction(o)).collect::<PyResult<_>>()?;
        macro_rules! go {
            ($($n:literal),*) => {
                match ops.len() {
                    $($n => {
                        let ops: [Reduction; $n] = ops.try_into().unwrap();
                        dispatch!(&self.inner, "reduce", [F32, F64, Df32, U32, I32], a => Inner::from(a.reduce(ops)))?
                    })*
                    _ => return Err(PyValueError::new_err("reduce takes one to four reductions")),
                }
            };
        }
        Ok(PyArr::new(go!(1, 2, 3, 4), 1))
    }

    /// Several reductions along one axis in one pass, as a tuple of arrays (axis removed unless
    /// `keepdims`).
    #[pyo3(signature = (axis, ops, keepdims = false))]
    fn reduce_axis(&self, axis: isize, ops: Vec<String>, keepdims: bool) -> PyResult<Vec<PyArr>> {
        let ops: Vec<Reduction> = ops.iter().map(|o| reduction(o)).collect::<PyResult<_>>()?;
        let la = luma_axis(axis, self.ndim)?;
        macro_rules! go {
            ($($n:literal),*) => {
                match ops.len() {
                    $($n => {
                        let ops: [Reduction; $n] = ops.try_into().unwrap();
                        dispatch!(&self.inner, "reduce_axis", [F32, F64, Df32, U32, I32], a => a.reduce_axis(la, ops).into_iter().map(Inner::from).collect::<Vec<_>>())?
                    })*
                    _ => return Err(PyValueError::new_err("reduce_axis takes one to four reductions")),
                }
            };
        }
        let results = go!(1, 2, 3, 4);
        let py_axis = norm_axis(axis, self.ndim)?;
        results
            .into_iter()
            .map(|r| {
                if keepdims {
                    Ok(PyArr::new(r, self.ndim))
                } else {
                    let mut shape = py_dims(r.shape(), self.ndim);
                    shape.remove(py_axis);
                    PyArr::new(r, 0).with_shape(&shape)
                }
            })
            .collect()
    }

    /// The sum of the elements where `cond` is true, in one pass.
    fn sum_where(&self, cond: &Bound<'_, PyAny>) -> PyResult<PyArr> {
        let (mask, _) = Mask::from_py(cond, self)?;
        let inner = with_mask!(&mask, m => dispatch!(&self.inner, "sum_where", [F32, F64, Df32, U32, I32], a => Inner::from(a.sum_where(m)))?);
        Ok(PyArr::new(inner, 0))
    }

    /// The flat index of the maximum (`axis=None`), or the index along `axis`; uint32.
    #[pyo3(signature = (axis = None, keepdims = false))]
    fn argmax(&self, axis: Option<&Bound<'_, PyAny>>, keepdims: bool) -> PyResult<PyArr> {
        self.reduce_with(
            axis,
            keepdims,
            false,
            &|x| dispatch!(x, "argmax", [F32, F64, Df32, U32, I32], a => Inner::from(a.argmax())),
            &|x, ax| dispatch!(x, "argmax", [F32, F64, Df32, U32, I32], a => Inner::from(a.argmax_axis(ax))),
        )
    }

    /// The flat index of the minimum (`axis=None`), or the index along `axis`; uint32.
    #[pyo3(signature = (axis = None, keepdims = false))]
    fn argmin(&self, axis: Option<&Bound<'_, PyAny>>, keepdims: bool) -> PyResult<PyArr> {
        self.reduce_with(
            axis,
            keepdims,
            false,
            &|x| dispatch!(x, "argmin", [F32, F64, Df32, U32, I32], a => Inner::from(a.argmin())),
            &|x, ax| dispatch!(x, "argmin", [F32, F64, Df32, U32, I32], a => Inner::from(a.argmin_axis(ax))),
        )
    }

    /// The sum of the elements that are not NaN (0 if all are).
    #[pyo3(signature = (axis = None, keepdims = false))]
    fn nansum(&self, axis: Option<&Bound<'_, PyAny>>, keepdims: bool) -> PyResult<PyArr> {
        self.reduce_with(
            axis,
            keepdims,
            false,
            &|x| dispatch!(x, "nansum", [F32, F64, Df32], a => Inner::from(a.nansum())),
            &|x, ax| Ok(f32_of(x, "nansum along an axis")?.nansum_axis(ax).into()),
        )
    }

    /// The minimum of the elements that are not NaN (NaN if all are).
    #[pyo3(signature = (axis = None, keepdims = false))]
    fn nanmin(&self, axis: Option<&Bound<'_, PyAny>>, keepdims: bool) -> PyResult<PyArr> {
        self.reduce_with(
            axis,
            keepdims,
            false,
            &|x| dispatch!(x, "nanmin", [F32, F64, Df32], a => Inner::from(a.nanmin())),
            &|x, ax| Ok(f32_of(x, "nanmin along an axis")?.nanmin_axis(ax).into()),
        )
    }

    /// The maximum of the elements that are not NaN (NaN if all are).
    #[pyo3(signature = (axis = None, keepdims = false))]
    fn nanmax(&self, axis: Option<&Bound<'_, PyAny>>, keepdims: bool) -> PyResult<PyArr> {
        self.reduce_with(
            axis,
            keepdims,
            false,
            &|x| dispatch!(x, "nanmax", [F32, F64, Df32], a => Inner::from(a.nanmax())),
            &|x, ax| Ok(f32_of(x, "nanmax along an axis")?.nanmax_axis(ax).into()),
        )
    }

    /// The mean of the elements that are not NaN; float32.
    #[pyo3(signature = (axis = None, keepdims = false))]
    fn nanmean(&self, axis: Option<&Bound<'_, PyAny>>, keepdims: bool) -> PyResult<PyArr> {
        self.reduce_with(axis, keepdims, false, &|x| Ok(f32_of(x, "nanmean")?.nanmean().into()), &|x, ax| {
            Ok(f32_of(x, "nanmean")?.nanmean_axis(ax).into())
        })
    }

    /// The `q`-quantile (`0 <= q <= 1`, linear interpolation as NumPy's default); float32.
    #[pyo3(signature = (q, axis = None, keepdims = false))]
    fn quantile(&self, q: f32, axis: Option<&Bound<'_, PyAny>>, keepdims: bool) -> PyResult<PyArr> {
        self.reduce_with(axis, keepdims, false, &|x| Ok(f32_of(x, "quantile")?.quantile(q).into()), &|x, ax| {
            Ok(f32_of(x, "quantile")?.quantile_axis(ax, q).into())
        })
    }

    /// The median; float32.
    #[pyo3(signature = (axis = None, keepdims = false))]
    fn median(&self, axis: Option<&Bound<'_, PyAny>>, keepdims: bool) -> PyResult<PyArr> {
        self.reduce_with(axis, keepdims, false, &|x| Ok(f32_of(x, "median")?.median().into()), &|x, ax| {
            Ok(f32_of(x, "median")?.median_axis(ax).into())
        })
    }

    /// The running sum along `axis` (the flattened array when `None`). `bool` gives uint32.
    #[pyo3(signature = (axis = None))]
    fn cumsum(&self, axis: Option<isize>) -> PyResult<PyArr> {
        let (src, ndim, la) = match axis {
            None => (self.flat(), 1, 0),
            Some(a) => (self.inner.clone(), self.ndim, luma_axis(a, self.ndim)?),
        };
        let r = dispatch!(&src, "cumsum", [F32, F64, Df32, U32, I32, Bool], a => Inner::from(a.cumsum(la)))?;
        Ok(PyArr::new(r, ndim))
    }

    // -- Sorting -----------------------------------------------------------------------------

    /// The values sorted along `axis` (default the last; `None` sorts the flattened array).
    /// `descending=True` works on one-dimensional (or flattened) arrays.
    #[pyo3(signature = (axis = Some(-1), descending = false), text_signature = "(self, /, axis=-1, descending=False)")]
    fn sort(&self, axis: Option<isize>, descending: bool) -> PyResult<PyArr> {
        let flat = axis.is_none() || self.ndim <= 1;
        if flat {
            let src = self.flat();
            let r = if descending {
                dispatch!(&src, "sort", [F32, U32, I32], a => Inner::from(a.sort_descending()))?
            } else {
                dispatch!(&src, "sort", [F32, U32, I32], a => Inner::from(a.sort()))?
            };
            let ndim = if axis.is_none() { 1 } else { self.ndim };
            return PyArr::new(r, 0).with_shape(&if ndim == 0 { vec![] } else { vec![self.inner.len()] });
        }
        if descending {
            return Err(PyValueError::new_err("descending sorts of a multi-dimensional array take axis=None"));
        }
        let la = luma_axis(axis.unwrap(), self.ndim)?;
        Ok(self.same(dispatch!(&self.inner, "sort", [F32, U32, I32], a => Inner::from(a.sort_axis(la)))?))
    }

    /// The indices that sort the flattened array (uint32); one-dimensional arrays only, or
    /// `axis=None`.
    #[pyo3(signature = (axis = Some(-1), descending = false), text_signature = "(self, /, axis=-1, descending=False)")]
    fn argsort(&self, axis: Option<isize>, descending: bool) -> PyResult<PyArr> {
        if axis.is_some() && self.ndim > 1 {
            return Err(PyValueError::new_err("argsort sorts the flattened array: pass axis=None"));
        }
        let src = self.flat();
        let r = if descending {
            dispatch!(&src, "argsort", [F32, U32, I32], a => Inner::from(a.argsort_descending()))?
        } else {
            dispatch!(&src, "argsort", [F32, U32, I32], a => Inner::from(a.argsort()))?
        };
        Ok(PyArr::new(r, 1))
    }

    /// The `k` largest values of the flattened array, largest first, and their flat indices.
    fn topk(&self, k: usize) -> PyResult<(PyArr, PyArr)> {
        let (v, i) = dispatch!(&self.inner, "topk", [F32, U32, I32], a => { let (v, i) = a.topk(k); (Inner::from(v), Inner::from(i)) })?;
        Ok((PyArr::new(v, 1), PyArr::new(i, 1)))
    }

    /// The `k` smallest values of the flattened array, smallest first, and their flat indices.
    fn bottomk(&self, k: usize) -> PyResult<(PyArr, PyArr)> {
        let (v, i) = dispatch!(&self.inner, "bottomk", [F32, U32, I32], a => { let (v, i) = a.bottomk(k); (Inner::from(v), Inner::from(i)) })?;
        Ok((PyArr::new(v, 1), PyArr::new(i, 1)))
    }

    /// `(uniques, codes)`: the sorted distinct values and, shaped like this array, each
    /// element's index into them. Synchronizes once.
    fn factorize(&self) -> PyResult<(PyArr, PyArr)> {
        let (u, c) = dispatch!(&self.inner, "factorize", [F32, U32, I32], a => { let (u, c) = a.factorize(); (Inner::from(u), Inner::from(c)) })?;
        Ok((PyArr::new(u, 1), self.same(c)))
    }

    /// The sorted distinct values. Synchronizes once.
    fn unique(&self) -> PyResult<PyArr> {
        Ok(PyArr::new(dispatch!(&self.inner, "unique", [F32, U32, I32], a => Inner::from(a.unique()))?, 1))
    }

    /// For this sorted 1-D array, the insertion index of every element of `values` (uint32,
    /// shaped like `values`), as `np.searchsorted`.
    #[pyo3(signature = (values, side = "left"))]
    fn searchsorted(&self, values: &Bound<'_, PyAny>, side: &str) -> PyResult<PyArr> {
        let side = match side {
            "left" => Side::Left,
            "right" => Side::Right,
            other => return Err(PyValueError::new_err(format!("side is \"left\" or \"right\", not {other:?}"))),
        };
        let v = same_dtype(values, self)?;
        let r = match (&self.flat(), &v.inner) {
            (Inner::F32(a), Inner::F32(b)) => Inner::from(a.searchsorted(b, side)),
            (Inner::U32(a), Inner::U32(b)) => Inner::from(a.searchsorted(b, side)),
            (Inner::I32(a), Inner::I32(b)) => Inner::from(a.searchsorted(b, side)),
            (other, _) => return Err(crate::inner::unsupported("searchsorted", other.dtype())),
        };
        Ok(PyArr::new(r, v.ndim))
    }

    /// The bin of every element for the sorted `bins` edges (uint32), as `np.digitize`.
    #[pyo3(signature = (bins, right = false))]
    fn digitize(&self, bins: &Bound<'_, PyAny>, right: bool) -> PyResult<PyArr> {
        let edges = same_dtype(bins, self)?;
        let r = match (&self.inner, &edges.inner) {
            (Inner::F32(a), Inner::F32(b)) => Inner::from(a.digitize(b, right)),
            (Inner::U32(a), Inner::U32(b)) => Inner::from(a.digitize(b, right)),
            (Inner::I32(a), Inner::I32(b)) => Inner::from(a.digitize(b, right)),
            (other, _) => return Err(crate::inner::unsupported("digitize", other.dtype())),
        };
        Ok(self.same(r))
    }

    /// Whether every element is in `test_elements` (bool, shaped like this array).
    fn isin(&self, test_elements: &Bound<'_, PyAny>) -> PyResult<PyArr> {
        let set = same_dtype(test_elements, self)?;
        let r = match (&self.inner, &set.inner) {
            (Inner::F32(a), Inner::F32(b)) => Inner::from(a.isin(b)),
            (Inner::U32(a), Inner::U32(b)) => Inner::from(a.isin(b)),
            (Inner::I32(a), Inner::I32(b)) => Inner::from(a.isin(b)),
            (other, _) => return Err(crate::inner::unsupported("isin", other.dtype())),
        };
        Ok(self.same(r))
    }

    // -- Group-by and histograms --------------------------------------------------------------

    /// Per group `g < groups`, the sum of the elements whose code is `g`.
    fn group_sum(&self, codes: &Bound<'_, PyAny>, groups: usize) -> PyResult<PyArr> {
        let (c, _) = indices(codes, self)?;
        Ok(PyArr::new(
            dispatch!(&self.inner, "group_sum", [F32, U32, I32], a => Inner::from(a.group_sum(&c, groups)))?,
            1,
        ))
    }

    /// Per group, the number of elements whose code is `g` (uint32).
    fn group_count(&self, codes: &Bound<'_, PyAny>, groups: usize) -> PyResult<PyArr> {
        let (c, _) = indices(codes, self)?;
        Ok(PyArr::new(
            dispatch!(&self.inner, "group_count", [F32, U32, I32], a => Inner::from(a.group_count(&c, groups)))?,
            1,
        ))
    }

    /// Per group, the minimum of the elements whose code is `g`.
    fn group_min(&self, codes: &Bound<'_, PyAny>, groups: usize) -> PyResult<PyArr> {
        let (c, _) = indices(codes, self)?;
        Ok(PyArr::new(
            dispatch!(&self.inner, "group_min", [F32, U32, I32], a => Inner::from(a.group_min(&c, groups)))?,
            1,
        ))
    }

    /// Per group, the maximum of the elements whose code is `g`.
    fn group_max(&self, codes: &Bound<'_, PyAny>, groups: usize) -> PyResult<PyArr> {
        let (c, _) = indices(codes, self)?;
        Ok(PyArr::new(
            dispatch!(&self.inner, "group_max", [F32, U32, I32], a => Inner::from(a.group_max(&c, groups)))?,
            1,
        ))
    }

    /// Per group, the mean of the elements whose code is `g` (float32).
    fn group_mean(&self, codes: &Bound<'_, PyAny>, groups: usize) -> PyResult<PyArr> {
        let (c, _) = indices(codes, self)?;
        Ok(PyArr::new(f32_of(&self.inner, "group_mean")?.group_mean(&c, groups).into(), 1))
    }

    /// The number of elements in each of `bins` equal bins over `[lo, hi)` (uint32); values
    /// outside are not counted.
    fn histogram(&self, bins: usize, lo: f32, hi: f32) -> PyResult<PyArr> {
        Ok(PyArr::new(f32_of(&self.inner, "histogram")?.histogram(bins, lo, hi).into(), 1))
    }

    // -- Selection -----------------------------------------------------------------------------

    /// The elements where `mask` holds, flattened; `None` if there are none. Synchronizes once
    /// to read the count; `filter_padded` stays on the device.
    fn filter(&self, py: Python<'_>, mask: &Bound<'_, PyAny>) -> PyResult<Option<PyArr>> {
        let (mask, _) = Mask::from_py(mask, self)?;
        let inner = &self.inner;
        let r = py.detach(|| {
            with_mask!(&mask, m => crate::dispatch_scalar!(inner, "filter", a => a.filter(m).map(|o| o.map(Inner::from))))
        })?;
        Ok(r.py()?.map(|i| PyArr::new(i, 1)))
    }

    /// `(values, count)`: the selected elements in flat order followed by `fill`, shaped like
    /// this array, and their number as a 0-d uint32 array. Nothing is read back.
    fn filter_padded(&self, mask: &Bound<'_, PyAny>, fill: &Bound<'_, PyAny>) -> PyResult<(PyArr, PyArr)> {
        let (mask, _) = Mask::from_py(mask, self)?;
        let fill = crate::dtype::PyNum::extract(fill).ok_or_else(|| PyTypeError::new_err("fill is a number"))?;
        use crate::dtype::Elem;
        macro_rules! go {
            ($($v:ident: $t:ty),*) => {
                match &self.inner {
                    $(Inner::$v(a) => with_mask!(&mask, m => { let (v, c) = a.filter_padded(m, <$t>::from_num(fill)?); (Inner::from(v), Inner::from(c)) }),)*
                }
            };
        }
        let (v, c) = go!(F32: f32, F64: f64, Df32: Df32, U32: u32, I32: i32, Bool: bool);
        Ok((self.same(v), PyArr::new(c, 0)))
    }

    /// Gathers by flat index: element `i` of the result is element `indices[i]` of the
    /// flattened array (clamped to the last). The result has the shape of `indices`.
    fn take(&self, indices: &Bound<'_, PyAny>) -> PyResult<PyArr> {
        let (idx, ndim) = self::indices(indices, self)?;
        Ok(PyArr::new(crate::dispatch_scalar!(&self.inner, "take", a => Inner::from(a.take(&idx)))?, ndim))
    }

    /// Scatters in place: `self.flat[indices[i]] = values[i]` (indices past the end are
    /// skipped). `values` is an array shaped like `indices` or a scalar.
    pub fn put(&mut self, indices: &Bound<'_, PyAny>, values: &Bound<'_, PyAny>) -> PyResult<()> {
        let (idx, _) = self::indices(indices, self)?;
        let rhs = values_rhs(values, self)?;
        mut_op!(&mut self.inner, rhs, "put", [F32: f32: full, U32: u32: full, I32: i32: full, Df32: Df32: val, Bool: bool: val, F64: f64: arr], a, b => a.put(&idx, b))
    }

    /// Scatters with addition in place: `self.flat[indices[i]] += values[i]`, equal indices
    /// all counting (atomics; float32 sums in no fixed order).
    fn put_add(&mut self, indices: &Bound<'_, PyAny>, values: &Bound<'_, PyAny>) -> PyResult<()> {
        let (idx, _) = self::indices(indices, self)?;
        let rhs = values_rhs(values, self)?;
        mut_op!(&mut self.inner, rhs, "put_add", [F32: f32: full, U32: u32: full, I32: i32: full], a, b => a.put_add(&idx, b))
    }

    /// Scatters with the minimum in place.
    fn put_min(&mut self, indices: &Bound<'_, PyAny>, values: &Bound<'_, PyAny>) -> PyResult<()> {
        let (idx, _) = self::indices(indices, self)?;
        let rhs = values_rhs(values, self)?;
        mut_op!(&mut self.inner, rhs, "put_min", [F32: f32: full, U32: u32: full, I32: i32: full], a, b => a.put_min(&idx, b))
    }

    /// Scatters with the maximum in place.
    fn put_max(&mut self, indices: &Bound<'_, PyAny>, values: &Bound<'_, PyAny>) -> PyResult<()> {
        let (idx, _) = self::indices(indices, self)?;
        let rhs = values_rhs(values, self)?;
        mut_op!(&mut self.inner, rhs, "put_max", [F32: f32: full, U32: u32: full, I32: i32: full], a, b => a.put_max(&idx, b))
    }

    /// 1-D convolution with `kernel` (float32), Luma's direct kernel.
    #[pyo3(signature = (kernel, stride = 1, padding = 0))]
    fn conv1d(&self, kernel: &Bound<'_, PyAny>, stride: usize, padding: usize) -> PyResult<PyArr> {
        let k = same_dtype(kernel, &self.as_f32()?)?;
        let k = f32_of(&k.inner, "conv1d")?;
        Ok(PyArr::new(f32_of(&self.inner, "conv1d")?.conv1d(&k, stride, padding).into(), 1))
    }

    /// 1-D convolution with `kernel` (float32) through the FFT, for long kernels.
    fn conv1d_fft(&self, kernel: &Bound<'_, PyAny>) -> PyResult<PyArr> {
        let k = same_dtype(kernel, &self.as_f32()?)?;
        let k = f32_of(&k.inner, "conv1d_fft")?;
        Ok(PyArr::new(f32_of(&self.inner, "conv1d_fft")?.conv1d_fft(&k).into(), 1))
    }
}
