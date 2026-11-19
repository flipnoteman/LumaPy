//! Host data in and out: NumPy arrays, lists and scalars.

use luma::{Array, Context, Df32};
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArrayDyn, PyReadwriteArrayDyn, PyUntypedArrayMethods};
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::array::PyArr;
use crate::dtype::DType;
use crate::error::IntoPy;
use crate::inner::{Inner, luma_shape, py_dims};

pub fn ctx_or_default(ctx: Option<&Context>) -> PyResult<Context> {
    match ctx {
        Some(c) => Ok(c.clone()),
        None => luma::default_context().py(),
    }
}

/// The dtype an array of NumPy dtype `name` becomes when none is asked for: floats become
/// `float32` and integers 32-bit, as GPU arrays usually are.
fn infer(kind: char, name: &str) -> PyResult<DType> {
    Ok(match kind {
        'b' => DType::Bool,
        'f' => DType::F32,
        'i' => DType::I32,
        'u' => DType::U32,
        _ => {
            return Err(PyTypeError::new_err(format!("cannot make a Luma array from NumPy dtype {name}")));
        }
    })
}

/// Any array-like (NumPy array, nested lists, scalar, `lumapy.Array`) as an array of `dtype`
/// (inferred when `None`) on `ctx` (the default context when `None`).
pub fn from_object(obj: &Bound<'_, PyAny>, dtype: Option<DType>, ctx: Option<&Context>) -> PyResult<PyArr> {
    if let Ok(arr) = obj.cast::<PyArr>() {
        let arr = arr.try_borrow()?;
        let inner = match dtype {
            Some(d) => arr.inner.astype(d)?,
            None => arr.inner.clone(),
        };
        return Ok(PyArr::new(inner, arr.ndim));
    }
    let py = obj.py();
    let np = py.import("numpy")?;
    let host = np.getattr("asarray")?.call1((obj,))?;
    let np_dtype = host.getattr("dtype")?;
    let kind: char = np_dtype.getattr("kind")?.extract()?;
    let name: String = np_dtype.getattr("name")?.extract()?;
    let dtype = match dtype {
        Some(d) => d,
        None => infer(kind, &name)?,
    };
    // Integers wider than 32 bits must fit: NumPy's cast would wrap them silently.
    if matches!(dtype, DType::I32 | DType::U32)
        && matches!(kind, 'i' | 'u')
        && host.getattr("size")?.extract::<usize>()? > 0
    {
        let (lo, hi) = if dtype == DType::I32 { (i32::MIN as i128, i32::MAX as i128) } else { (0, u32::MAX as i128) };
        let min: i128 = host.call_method0("min")?.call_method0("item")?.extract()?;
        let max: i128 = host.call_method0("max")?.call_method0("item")?.extract()?;
        if min < lo || max > hi {
            return Err(PyValueError::new_err(format!("values in [{min}, {max}] do not fit {}", dtype.name())));
        }
    }
    // Before `ascontiguousarray`, which makes 0-d arrays 1-d.
    let shape: Vec<usize> = host.getattr("shape")?.extract()?;
    let host = np.getattr("ascontiguousarray")?.call1((host, dtype.numpy_name()))?;
    let lshape = luma_shape(&shape)?;
    let ctx = ctx_or_default(ctx)?;

    macro_rules! upload {
        ($t:ty) => {{
            let data: PyReadonlyArrayDyn<'_, $t> = host.extract()?;
            let slice = data.as_slice()?;
            Inner::from(py.detach(|| Array::<$t>::from_slice_on(&ctx, slice, lshape)).py()?)
        }};
    }
    let inner = match dtype {
        DType::F32 => upload!(f32),
        DType::F64 => upload!(f64),
        DType::U32 => upload!(u32),
        DType::I32 => upload!(i32),
        DType::Bool => upload!(bool),
        DType::Df32 => {
            let data: PyReadonlyArrayDyn<'_, f64> = host.extract()?;
            let values: Vec<Df32> = data.as_slice()?.iter().map(|&v| Df32::from(v)).collect();
            Inner::from(py.detach(|| Array::from_slice_on(&ctx, &values, lshape)).py()?)
        }
    };
    Ok(PyArr::new(inner, shape.len()))
}

/// The array's values as a new NumPy array (C order, NumPy-order shape); waits for the GPU.
pub fn to_numpy<'py>(py: Python<'py>, inner: &Inner, ndim: usize) -> PyResult<Bound<'py, PyAny>> {
    let dims = py_dims(inner.shape(), ndim);
    macro_rules! download {
        ($a:expr) => {{
            let values = py.detach(|| $a.to_vec()).py()?;
            PyArray1::from_vec(py, values).reshape(dims)?.into_any()
        }};
    }
    Ok(match inner {
        Inner::F32(a) => download!(a),
        Inner::F64(a) => download!(a),
        Inner::U32(a) => download!(a),
        Inner::I32(a) => download!(a),
        Inner::Bool(a) => download!(a),
        Inner::Df32(a) => {
            let values = py.detach(|| a.to_vec()).py()?;
            let values: Vec<f64> = values.into_iter().map(Df32::to_f64).collect();
            PyArray1::from_vec(py, values).reshape(dims)?.into_any()
        }
    })
}

/// Reads the array into an existing C-contiguous NumPy array of the same dtype and size
/// (Luma's `read_into`, without allocating).
pub fn read_into(py: Python<'_>, inner: &Inner, out: &Bound<'_, PyAny>) -> PyResult<()> {
    macro_rules! read {
        ($a:expr, $t:ty) => {{
            let mut target: PyReadwriteArrayDyn<'_, $t> = out.extract().map_err(|_| {
                PyTypeError::new_err(format!(
                    "read_into needs a writable NumPy array of dtype {}",
                    inner.dtype().numpy_name()
                ))
            })?;
            if target.len() != inner.len() {
                return Err(PyValueError::new_err(format!(
                    "read_into: the array has {} elements, the target {}",
                    inner.len(),
                    target.len()
                )));
            }
            let slice = target.as_slice_mut()?;
            py.detach(|| $a.read_into(slice)).py()
        }};
    }
    match inner {
        Inner::F32(a) => read!(a, f32),
        Inner::F64(a) => read!(a, f64),
        Inner::U32(a) => read!(a, u32),
        Inner::I32(a) => read!(a, i32),
        Inner::Bool(a) => read!(a, bool),
        Inner::Df32(_) => Err(PyTypeError::new_err("read_into does not take df32 arrays; use to_numpy()")),
    }
}

/// Overwrites the array's data with host values of the same number of elements.
pub fn write(py: Python<'_>, inner: &mut Inner, data: &Bound<'_, PyAny>) -> PyResult<()> {
    let np = py.import("numpy")?;
    let host = np.getattr("ascontiguousarray")?.call1((data, inner.dtype().numpy_name()))?;
    let size: usize = host.getattr("size")?.extract()?;
    if size != inner.len() {
        return Err(PyValueError::new_err(format!("write: the array has {} elements, the data {size}", inner.len())));
    }
    macro_rules! put {
        ($a:expr, $t:ty) => {{
            let host: PyReadonlyArrayDyn<'_, $t> = host.extract()?;
            let slice = host.as_slice()?;
            py.detach(|| $a.write(slice)).py()
        }};
    }
    match inner {
        Inner::F32(a) => put!(a, f32),
        Inner::F64(a) => put!(a, f64),
        Inner::U32(a) => put!(a, u32),
        Inner::I32(a) => put!(a, i32),
        Inner::Bool(a) => put!(a, bool),
        Inner::Df32(a) => {
            let host: PyReadonlyArrayDyn<'_, f64> = host.extract()?;
            let values: Vec<Df32> = host.as_slice()?.iter().map(|&v| Df32::from(v)).collect();
            py.detach(|| a.write(&values)).py()
        }
    }
}

/// The single value of a one-element array as a Python scalar; waits for the GPU.
pub fn item(py: Python<'_>, inner: &Inner) -> PyResult<Py<PyAny>> {
    if inner.len() != 1 {
        return Err(PyValueError::new_err(format!(
            "item() needs an array of one element, this one has {}",
            inner.len()
        )));
    }
    match inner {
        Inner::F32(a) => py.detach(|| a.item()).py()?.into_py_any(py),
        Inner::F64(a) => py.detach(|| a.item()).py()?.into_py_any(py),
        Inner::U32(a) => py.detach(|| a.item()).py()?.into_py_any(py),
        Inner::I32(a) => py.detach(|| a.item()).py()?.into_py_any(py),
        Inner::Bool(a) => py.detach(|| a.item()).py()?.into_py_any(py),
        Inner::Df32(a) => py.detach(|| a.item()).py()?.to_f64().into_py_any(py),
    }
}

/// A one-element array of `value` as `dtype`, broadcastable against any shape.
pub fn full(ctx: &Context, shape: luma::Shape, dtype: DType, value: crate::dtype::PyNum) -> PyResult<Inner> {
    use crate::dtype::Elem;
    Ok(match dtype {
        DType::F32 => Array::full_on(ctx, shape, f32::from_num(value)?).py()?.into(),
        DType::F64 => Array::full_on(ctx, shape, f64::from_num(value)?).py()?.into(),
        DType::Df32 => Array::full_on(ctx, shape, Df32::from_num(value)?).py()?.into(),
        DType::U32 => Array::full_on(ctx, shape, u32::from_num(value)?).py()?.into(),
        DType::I32 => Array::full_on(ctx, shape, i32::from_num(value)?).py()?.into(),
        DType::Bool => Array::full_on(ctx, shape, bool::from_num(value)?).py()?.into(),
    })
}
