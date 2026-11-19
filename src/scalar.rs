//! `lumapy.Scalar`: a value read by kernels from a uniform, changeable without recompiling.

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use crate::dtype::{DType, Elem, PyNum};

#[derive(Clone, Debug)]
pub enum ScalarInner {
    F32(luma::Scalar<f32>),
    U32(luma::Scalar<u32>),
    I32(luma::Scalar<i32>),
}

impl ScalarInner {
    pub fn dtype(&self) -> DType {
        match self {
            ScalarInner::F32(_) => DType::F32,
            ScalarInner::U32(_) => DType::U32,
            ScalarInner::I32(_) => DType::I32,
        }
    }
}

/// A 32-bit value (`float32`, `uint32` or `int32`) that array operations read at run time, so
/// changing it with `set` does not compile a new kernel. Operations recorded after `set` see
/// the new value; those recorded before keep the old one.
///
/// ```python
/// s = lumapy.Scalar(2.0)
/// a = x * s      # doubles
/// s.set(3.0)
/// b = x * s      # triples; `a` still doubles
/// ```
#[pyclass(name = "Scalar", module = "lumapy", frozen)]
pub struct PyScalar {
    pub inner: ScalarInner,
}

#[pymethods]
impl PyScalar {
    #[new]
    #[pyo3(signature = (value, dtype = None))]
    fn new(value: &Bound<'_, PyAny>, dtype: Option<&Bound<'_, PyAny>>) -> PyResult<PyScalar> {
        let num = PyNum::extract(value).ok_or_else(|| PyTypeError::new_err("Scalar takes a number"))?;
        let dtype = match dtype {
            Some(d) => DType::from_py(d)?,
            None => num.natural_dtype(),
        };
        let inner = match dtype {
            DType::F32 => ScalarInner::F32(luma::Scalar::new(f32::from_num(num)?)),
            DType::U32 => ScalarInner::U32(luma::Scalar::new(u32::from_num(num)?)),
            DType::I32 => ScalarInner::I32(luma::Scalar::new(i32::from_num(num)?)),
            other => {
                return Err(PyTypeError::new_err(format!(
                    "a Scalar is float32, uint32 or int32, not {}",
                    other.name()
                )));
            }
        };
        Ok(PyScalar { inner })
    }

    /// The current value.
    fn get(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        use pyo3::IntoPyObjectExt;
        match &self.inner {
            ScalarInner::F32(s) => s.get().into_py_any(py),
            ScalarInner::U32(s) => s.get().into_py_any(py),
            ScalarInner::I32(s) => s.get().into_py_any(py),
        }
    }

    /// Changes the value for operations recorded from now on.
    fn set(&self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let num = PyNum::extract(value).ok_or_else(|| PyTypeError::new_err("Scalar.set takes a number"))?;
        match &self.inner {
            ScalarInner::F32(s) => s.set(f32::from_num(num)?),
            ScalarInner::U32(s) => s.set(u32::from_num(num)?),
            ScalarInner::I32(s) => s.set(i32::from_num(num)?),
        }
        Ok(())
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().name()
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!("lumapy.Scalar({}, dtype={})", self.get(py)?.bind(py).repr()?, self.dtype()))
    }
}
