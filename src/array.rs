//! `lumapy.Array`: core attributes, host transfer, casts and evaluation. Operators, reductions
//! and views are in their own `#[pymethods]` blocks (`ops.rs`, `reduce.rs`, `view.rs`).

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::context::PyContext;
use crate::convert;
use crate::dtype::DType;
use crate::each;
use crate::error::IntoPy;
use crate::inner::{Inner, luma_shape, py_dims, rank};

/// A lazy array on the GPU, of up to four dimensions.
///
/// Shapes and axes follow NumPy: the last axis varies fastest, `a @ b` is the matrix product,
/// `axis=0` reduces over rows. Operations are recorded, fused into kernels and batched; nothing
/// waits for the GPU until the host needs values (`to_numpy`, `item`, `tolist`, `repr`).
///
/// A failed operation does not raise where it is written: it yields a *poisoned* array, and
/// the `LumaError` (with the failed operation and its cause) is raised where a result is needed.
/// `error` and `check()` inspect it without reading the data.
#[pyclass(name = "Array", module = "lumapy")]
pub struct PyArr {
    pub inner: Inner,
    pub ndim: usize,
}

impl PyArr {
    /// An array with at least `ndim` NumPy axes (more if the shape needs them).
    pub fn new(inner: Inner, ndim: usize) -> PyArr {
        let ndim = ndim.max(rank(inner.shape()));
        PyArr { inner, ndim }
    }

    pub fn py_shape(&self) -> Vec<usize> {
        py_dims(self.inner.shape(), self.ndim)
    }

    /// Another wrapper on the same data with a different NumPy shape.
    pub fn with_shape(&self, shape: &[usize]) -> PyResult<PyArr> {
        Ok(PyArr::new(self.inner.reshape(luma_shape(shape)?), shape.len()))
    }

    pub fn same(&self, inner: Inner) -> PyArr {
        PyArr::new(inner, self.ndim)
    }
}

#[pymethods]
impl PyArr {
    /// `Array(obj, dtype=None, ctx=None)` is `lumapy.asarray(obj, dtype, ctx)`.
    #[new]
    #[pyo3(signature = (obj, dtype = None, ctx = None))]
    fn py_new(
        obj: &Bound<'_, PyAny>,
        dtype: Option<&Bound<'_, PyAny>>,
        ctx: Option<PyRef<'_, PyContext>>,
    ) -> PyResult<PyArr> {
        let dtype = dtype.map(DType::from_py).transpose()?;
        convert::from_object(obj, dtype, ctx.as_deref().map(|c| &c.inner))
    }

    /// The NumPy-order shape.
    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.py_shape())
    }

    #[getter]
    fn ndim(&self) -> usize {
        self.ndim
    }

    /// The number of elements.
    #[getter]
    fn size(&self) -> usize {
        self.inner.len()
    }

    /// `float32`, `float64`, `df32`, `uint32`, `int32` or `bool`.
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().name()
    }

    /// Bytes per element on the host side.
    #[getter]
    fn itemsize(&self) -> usize {
        match self.inner.dtype() {
            DType::F64 | DType::Df32 => 8,
            DType::Bool => 1,
            _ => 4,
        }
    }

    /// The `Context` (device) the array lives on.
    #[getter]
    fn context(&self) -> PyContext {
        PyContext { inner: self.inner.context() }
    }

    /// The shape as Luma sees it: four extents, the fastest axis first.
    #[getter]
    fn luma_shape(&self) -> [usize; 4] {
        self.inner.shape().dims()
    }

    fn __len__(&self) -> PyResult<usize> {
        self.py_shape().first().copied().ok_or_else(|| PyTypeError::new_err("len() of a 0-d array"))
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        if let Some(err) = self.error() {
            return Ok(format!("lumapy.Array(<poisoned: {err}>, dtype={})", self.dtype()));
        }
        let host = convert::to_numpy(py, &self.inner, self.ndim)?;
        let np = py.import("numpy")?;
        let body: String = np.getattr("array2string")?.call1((host,))?.extract()?;
        let body = body.replace('\n', "\n              ");
        Ok(format!("lumapy.Array({body}, dtype={})", self.dtype()))
    }

    fn __str__(&self, py: Python<'_>) -> PyResult<String> {
        let host = convert::to_numpy(py, &self.inner, self.ndim)?;
        host.str()?.extract()
    }

    fn __bool__(&self, py: Python<'_>) -> PyResult<bool> {
        if self.inner.len() != 1 {
            return Err(PyValueError::new_err(
                "the truth value of an array with more than one element is ambiguous; use .any() or .all()",
            ));
        }
        convert::item(py, &self.inner)?.bind(py).is_truthy()
    }

    fn __float__(&self, py: Python<'_>) -> PyResult<f64> {
        convert::item(py, &self.inner)?.extract(py)
    }

    fn __int__(&self, py: Python<'_>) -> PyResult<i64> {
        let value = convert::item(py, &self.inner)?;
        let value = value.bind(py);
        if let Ok(i) = value.extract::<i64>() {
            return Ok(i);
        }
        Ok(value.extract::<f64>()? as i64)
    }

    fn __index__(&self, py: Python<'_>) -> PyResult<i64> {
        if self.inner.dtype().is_float() {
            return Err(PyTypeError::new_err("only integer arrays can be used as an index"));
        }
        self.__int__(py)
    }

    /// The values as a new NumPy array; waits for the GPU. `df32` arrays come back as `float64`.
    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        convert::to_numpy(py, &self.inner, self.ndim)
    }

    /// NumPy's conversion protocol: `np.asarray(x)` reads the array.
    #[pyo3(signature = (dtype = None, copy = None))]
    fn __array__<'py>(
        &self,
        py: Python<'py>,
        dtype: Option<&Bound<'py, PyAny>>,
        copy: Option<bool>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if copy == Some(false) {
            return Err(PyValueError::new_err("a lumapy.Array cannot be viewed by NumPy without a copy"));
        }
        let host = convert::to_numpy(py, &self.inner, self.ndim)?;
        match dtype {
            Some(d) => host.call_method1("astype", (d,)),
            None => Ok(host),
        }
    }

    /// NumPy defers its operators to this class (`ndarray + Array` runs on the GPU).
    #[classattr]
    fn __array_ufunc__(py: Python<'_>) -> Py<PyAny> {
        py.None()
    }

    /// The values as (nested) Python lists; waits for the GPU.
    fn tolist<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.to_numpy(py)?.call_method0("tolist")
    }

    /// The value of a one-element array as a Python scalar; waits for the GPU.
    fn item(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        convert::item(py, &self.inner)
    }

    /// Reads the array into an existing NumPy array of the same dtype and size.
    fn read_into(&self, py: Python<'_>, out: &Bound<'_, PyAny>) -> PyResult<()> {
        convert::read_into(py, &self.inner, out)
    }

    /// Overwrites the data in place with host values of the same size (every handle on the
    /// data sees the change).
    fn write(&mut self, py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<()> {
        convert::write(py, &mut self.inner, data)
    }

    /// Starts computing the array without waiting for it (the GPU runs while Python records
    /// more work). Raises if the array is poisoned. Returns the array.
    fn eval(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Py<PyArr>> {
        let inner = slf.inner.clone();
        py.detach(|| each!(&inner, a => a.eval())).py()?;
        Ok(slf.into())
    }

    /// The error a poisoned array carries, as a message; `None` if it is not poisoned. Does not
    /// wait for the GPU.
    #[getter]
    fn error(&self) -> Option<String> {
        each!(&self.inner, a => a.error().map(|e| e.to_string()))
    }

    /// Raises the array's `LumaError` if it is poisoned; does not wait for the GPU.
    fn check(&self) -> PyResult<()> {
        each!(&self.inner, a => a.check()).py()
    }

    /// An independent copy of the data (a plain `Array(x)` or assignment shares it).
    fn copy(&self) -> PyArr {
        self.same(each!(&self.inner, a => a.copy().into()))
    }

    /// The array converted to `dtype`. Float to integer truncates toward zero; anything to
    /// `bool` is `!= 0`.
    fn astype(&self, dtype: &Bound<'_, PyAny>) -> PyResult<PyArr> {
        Ok(self.same(self.inner.astype(DType::from_py(dtype)?)?))
    }

    fn to_f32(&self) -> PyResult<PyArr> {
        Ok(self.same(self.inner.astype(DType::F32)?))
    }

    fn to_f64(&self) -> PyResult<PyArr> {
        Ok(self.same(self.inner.astype(DType::F64)?))
    }

    fn to_df32(&self) -> PyResult<PyArr> {
        Ok(self.same(self.inner.astype(DType::Df32)?))
    }

    fn to_u32(&self) -> PyResult<PyArr> {
        Ok(self.same(self.inner.astype(DType::U32)?))
    }

    fn to_i32(&self) -> PyResult<PyArr> {
        Ok(self.same(self.inner.astype(DType::I32)?))
    }

    /// Computes the array into a buffer of its own now (recorded, not waited for), so the
    /// expressions reading it read the buffer instead of recomputing it.
    fn cache(&self) -> PyResult<PyArr> {
        Ok(self.same(crate::dispatch_scalar!(&self.inner, "cache", a => a.cache().into())?))
    }
}
