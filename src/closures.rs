//! Operations that take a Python function: `iterate`, `iterate_until`, `scan`, `iterate_all`.
//!
//! Luma traces the function into a kernel: it is called with lazy arrays and must return lazy
//! arrays built from them, not read values back. An exception raised inside it is re-raised once
//! Luma returns.

use std::cell::RefCell;

use luma::{Array, Df32};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::array::PyArr;
use crate::dtype::{DType, Elem};
use crate::inner::{Inner, luma_axis};
use crate::operand::common;

/// Calls `f(*args)` and reads `n_out` arrays of `T` back; on failure stores the error and
/// returns the first `n_out` arguments so that Luma can finish.
fn call<T: Elem>(
    f: &Bound<'_, PyAny>,
    args: &[Array<T>],
    ndim: usize,
    n_out: usize,
    err: &RefCell<Option<PyErr>>,
) -> Vec<Array<T>> {
    let fallback = || args.iter().take(n_out).cloned().collect::<Vec<_>>();
    if err.borrow().is_some() {
        return fallback();
    }
    let py = f.py();
    let result = (|| -> PyResult<Vec<Array<T>>> {
        let wrapped =
            args.iter().map(|a| Py::new(py, PyArr::new(T::wrap(a.clone()), ndim))).collect::<PyResult<Vec<_>>>()?;
        let out = f.call1(PyTuple::new(py, wrapped)?)?;
        let outs: Vec<Bound<'_, PyAny>> =
            if out.cast::<PyArr>().is_ok() { vec![out] } else { out.try_iter()?.collect::<PyResult<_>>()? };
        if outs.len() != n_out {
            return Err(PyValueError::new_err(format!(
                "the function returned {} arrays, expected {n_out}",
                outs.len()
            )));
        }
        outs.iter()
            .map(|o| {
                let a = o
                    .cast::<PyArr>()
                    .map_err(|_| PyTypeError::new_err("the function must return lumapy arrays"))?
                    .borrow();
                T::take(a.inner.astype(T::DTYPE)?).ok_or_else(|| PyTypeError::new_err("unexpected dtype"))
            })
            .collect()
    })();
    match result {
        Ok(v) => v,
        Err(e) => {
            *err.borrow_mut() = Some(e);
            fallback()
        }
    }
}

fn finish<R>(value: R, err: RefCell<Option<PyErr>>) -> PyResult<R> {
    match err.into_inner() {
        Some(e) => Err(e),
        None => Ok(value),
    }
}

fn scan_t<T: Elem + luma::Numeric>(a: &Array<T>, axis: usize, f: &Bound<'_, PyAny>, ndim: usize) -> PyResult<Inner> {
    let err = RefCell::new(None);
    let r = a.scan(axis, |x, y| call(f, &[x.clone(), y.clone()], ndim, 1, &err).remove(0));
    finish(T::wrap(r), err)
}

fn iterate_all_t<T: Elem, const N: usize>(
    init: &[Array<T>],
    steps: usize,
    f: &Bound<'_, PyAny>,
    ndim: usize,
) -> PyResult<Vec<Inner>> {
    let err = RefCell::new(None);
    let refs: [&Array<T>; N] = std::array::from_fn(|i| &init[i]);
    let out = luma::iterate_all(refs, steps, |state: &[Array<T>; N]| {
        let v = call(f, state, ndim, N, &err);
        v.try_into().unwrap_or_else(|_| state.clone())
    });
    finish(out.into_iter().map(T::wrap).collect(), err)
}

fn iterate_all_n<T: Elem>(init: &[Array<T>], steps: usize, f: &Bound<'_, PyAny>, ndim: usize) -> PyResult<Vec<Inner>> {
    match init.len() {
        1 => iterate_all_t::<T, 1>(init, steps, f, ndim),
        2 => iterate_all_t::<T, 2>(init, steps, f, ndim),
        3 => iterate_all_t::<T, 3>(init, steps, f, ndim),
        4 => iterate_all_t::<T, 4>(init, steps, f, ndim),
        5 => iterate_all_t::<T, 5>(init, steps, f, ndim),
        6 => iterate_all_t::<T, 6>(init, steps, f, ndim),
        7 => iterate_all_t::<T, 7>(init, steps, f, ndim),
        8 => iterate_all_t::<T, 8>(init, steps, f, ndim),
        n => Err(PyValueError::new_err(format!("iterate_all takes 1 to 8 arrays, got {n}"))),
    }
}

/// Runs `steps` steps of `fn(*state) -> state` over several arrays, compiled into one kernel
/// that keeps the state in registers (elementwise steps only). Returns the final arrays.
#[pyfunction]
pub fn iterate_all(init: Vec<PyRef<'_, PyArr>>, steps: usize, f: &Bound<'_, PyAny>) -> PyResult<Vec<PyArr>> {
    let ndim = init.iter().map(|a| a.ndim).max().unwrap_or(0);
    let dtype = init.iter().try_fold(None, |acc: Option<DType>, a| -> PyResult<_> {
        Ok(Some(match acc {
            None => a.inner.dtype(),
            Some(d) => common(d, a.inner.dtype())?,
        }))
    })?;
    let Some(dtype) = dtype else {
        return Err(PyValueError::new_err("iterate_all takes at least one array"));
    };
    let inners: Vec<Inner> = init.iter().map(|a| a.inner.astype(dtype)).collect::<PyResult<_>>()?;
    macro_rules! go {
        ($t:ty) => {{
            let typed: Vec<Array<$t>> = inners.into_iter().map(|i| <$t>::take(i).unwrap()).collect();
            iterate_all_n::<$t>(&typed, steps, f, ndim)?
        }};
    }
    let out = match dtype {
        DType::F32 => go!(f32),
        DType::F64 => go!(f64),
        DType::Df32 => go!(Df32),
        DType::U32 => go!(u32),
        DType::I32 => go!(i32),
        DType::Bool => go!(bool),
    };
    Ok(out.into_iter().map(|i| PyArr::new(i, ndim)).collect())
}

#[pymethods]
impl PyArr {
    /// Runs `steps` steps of `fn(x) -> x` (float32), compiled into multi-step kernels where
    /// the step is a stencil. Returns the final array.
    fn iterate(&self, steps: usize, f: &Bound<'_, PyAny>) -> PyResult<PyArr> {
        let Inner::F32(a) = self.inner.astype(DType::F32)? else { unreachable!() };
        let err = RefCell::new(None);
        let r = a.iterate(steps, |x| call(f, std::slice::from_ref(x), self.ndim, 1, &err).remove(0));
        Ok(self.same(finish(r, err)?.into()))
    }

    /// Steps `fn(x) -> x` (float32) until the largest change of an element is below `tol`, or
    /// `max_steps` steps; the condition is checked on the device.
    fn iterate_until(&self, max_steps: usize, tol: f32, f: &Bound<'_, PyAny>) -> PyResult<PyArr> {
        let Inner::F32(a) = self.inner.astype(DType::F32)? else { unreachable!() };
        let err = RefCell::new(None);
        let r = a.iterate_until(max_steps, tol, |x| call(f, std::slice::from_ref(x), self.ndim, 1, &err).remove(0));
        Ok(self.same(finish(r, err)?.into()))
    }

    /// The inclusive scan along `axis` with the associative `fn(a, b)` (the flattened array
    /// when `axis` is `None`).
    #[pyo3(signature = (f, axis = None))]
    fn scan(&self, f: &Bound<'_, PyAny>, axis: Option<isize>) -> PyResult<PyArr> {
        let (src, ndim, la) = match axis {
            None => (self.flat(), 1, 0),
            Some(a) => (self.inner.clone(), self.ndim, luma_axis(a, self.ndim)?),
        };
        let r = match &src {
            Inner::F32(a) => scan_t(a, la, f, ndim)?,
            Inner::F64(a) => scan_t(a, la, f, ndim)?,
            Inner::Df32(a) => scan_t(a, la, f, ndim)?,
            Inner::U32(a) => scan_t(a, la, f, ndim)?,
            Inner::I32(a) => scan_t(a, la, f, ndim)?,
            Inner::Bool(_) => return Err(crate::inner::unsupported("scan", DType::Bool)),
        };
        Ok(PyArr::new(r, ndim))
    }
}
