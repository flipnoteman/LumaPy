//! Element types and their Python names.

use luma::{Array, Df32, ScalarType};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyString, PyType};

use crate::inner::Inner;
use crate::scalar::ScalarInner;

/// The element type of an array.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F64,
    Df32,
    U32,
    I32,
    Bool,
}

impl DType {
    /// The name Python sees (`Array.dtype`), NumPy's where one exists.
    pub fn name(self) -> &'static str {
        match self {
            DType::F32 => "float32",
            DType::F64 => "float64",
            DType::Df32 => "df32",
            DType::U32 => "uint32",
            DType::I32 => "int32",
            DType::Bool => "bool",
        }
    }

    /// The NumPy dtype name host data of this type is exchanged as (`df32` travels as `float64`).
    pub fn numpy_name(self) -> &'static str {
        match self {
            DType::Df32 => "float64",
            other => other.name(),
        }
    }

    pub fn is_float(self) -> bool {
        matches!(self, DType::F32 | DType::F64 | DType::Df32)
    }

    pub fn parse(name: &str) -> PyResult<DType> {
        Ok(match name.to_ascii_lowercase().as_str() {
            "float32" | "f32" | "float" | "single" => DType::F32,
            "float64" | "f64" | "double" => DType::F64,
            "df32" | "double_float" => DType::Df32,
            "uint32" | "u32" | "uint" => DType::U32,
            "int32" | "i32" | "int" => DType::I32,
            "bool" | "bool_" | "b1" => DType::Bool,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unsupported dtype {other:?}: Luma arrays hold float32, float64, df32, uint32, int32 or bool"
                )));
            }
        })
    }

    /// A dtype from what Python code passes: a name, a Python type (`float`, `int`, `bool`) or
    /// anything NumPy understands as a dtype (`np.float32`, `np.dtype("int32")`).
    pub fn from_py(obj: &Bound<'_, PyAny>) -> PyResult<DType> {
        if let Ok(name) = obj.cast::<PyString>() {
            return DType::parse(&name.to_cow()?);
        }
        if let Ok(ty) = obj.cast::<PyType>() {
            let py = obj.py();
            if ty.is(py.get_type::<pyo3::types::PyBool>()) {
                return Ok(DType::Bool);
            }
            if ty.is(py.get_type::<pyo3::types::PyInt>()) {
                return Ok(DType::I32);
            }
            if ty.is(py.get_type::<pyo3::types::PyFloat>()) {
                return Ok(DType::F32);
            }
        }
        // NumPy dtypes and scalar types: ask NumPy for the canonical name.
        let numpy = obj.py().import("numpy")?;
        let name: String = numpy.getattr("dtype")?.call1((obj,))?.getattr("name")?.extract()?;
        DType::parse(&name)
    }
}

/// A Python scalar on its way into an array operation.
#[derive(Clone, Copy, Debug)]
pub enum PyNum {
    Bool(bool),
    Int(i64),
    Float(f64),
}

impl PyNum {
    /// Reads a Python (or NumPy) scalar; `None` if `obj` is not one.
    pub fn extract(obj: &Bound<'_, PyAny>) -> Option<PyNum> {
        if let Ok(b) = obj.extract::<bool>() {
            return Some(PyNum::Bool(b));
        }
        if obj.is_instance_of::<pyo3::types::PyFloat>() {
            return obj.extract::<f64>().ok().map(PyNum::Float);
        }
        if let Ok(i) = obj.extract::<i64>() {
            return Some(PyNum::Int(i));
        }
        obj.extract::<f64>().ok().map(PyNum::Float)
    }

    /// The dtype a scalar has on its own (no array to take the type from).
    pub fn natural_dtype(self) -> DType {
        match self {
            PyNum::Bool(_) => DType::Bool,
            PyNum::Int(_) => DType::I32,
            PyNum::Float(_) => DType::F32,
        }
    }
}

/// The element types an `Inner` can hold, with the conversions the bindings need.
pub trait Elem: ScalarType + Send + Sync {
    const DTYPE: DType;

    /// The array, if `inner` holds this element type.
    fn take(inner: Inner) -> Option<Array<Self>>;

    /// The array as an `Inner`.
    fn wrap(array: Array<Self>) -> Inner;

    /// `bool` arrays promote to every type (`true` is 1), as in Luma's mixed operators.
    fn from_bool(array: &Array<bool>) -> Array<Self>;

    /// A Python scalar as this type; floats do not silently become integers.
    fn from_num(num: PyNum) -> PyResult<Self>;

    /// A Python-side `Scalar` of this type.
    fn scalar(_s: &ScalarInner) -> Option<luma::Scalar<Self>> {
        None
    }
}

fn int_range<T: TryFrom<i64>>(v: i64, dtype: DType) -> PyResult<T> {
    T::try_from(v).map_err(|_| PyValueError::new_err(format!("{v} is out of range for {}", dtype.name())))
}

fn float_into_int(dtype: DType) -> PyErr {
    PyTypeError::new_err(format!(
        "a float scalar cannot combine with a {} array; convert the array with .astype(\"float32\") first",
        dtype.name()
    ))
}

impl Elem for f32 {
    const DTYPE: DType = DType::F32;
    fn wrap(array: Array<f32>) -> Inner {
        Inner::F32(array)
    }
    fn take(inner: Inner) -> Option<Array<f32>> {
        match inner {
            Inner::F32(a) => Some(a),
            _ => None,
        }
    }
    fn from_bool(array: &Array<bool>) -> Array<f32> {
        array.to_f32()
    }
    fn from_num(num: PyNum) -> PyResult<f32> {
        Ok(match num {
            PyNum::Bool(b) => f32::from(u8::from(b)),
            PyNum::Int(i) => i as f32,
            PyNum::Float(f) => f as f32,
        })
    }
    fn scalar(s: &ScalarInner) -> Option<luma::Scalar<f32>> {
        match s {
            ScalarInner::F32(s) => Some(s.clone()),
            _ => None,
        }
    }
}

impl Elem for f64 {
    const DTYPE: DType = DType::F64;
    fn wrap(array: Array<f64>) -> Inner {
        Inner::F64(array)
    }
    fn take(inner: Inner) -> Option<Array<f64>> {
        match inner {
            Inner::F64(a) => Some(a),
            _ => None,
        }
    }
    fn from_bool(array: &Array<bool>) -> Array<f64> {
        array.to_f64()
    }
    fn from_num(num: PyNum) -> PyResult<f64> {
        Ok(match num {
            PyNum::Bool(b) => f64::from(u8::from(b)),
            PyNum::Int(i) => i as f64,
            PyNum::Float(f) => f,
        })
    }
}

impl Elem for Df32 {
    const DTYPE: DType = DType::Df32;
    fn wrap(array: Array<Df32>) -> Inner {
        Inner::Df32(array)
    }
    fn take(inner: Inner) -> Option<Array<Df32>> {
        match inner {
            Inner::Df32(a) => Some(a),
            _ => None,
        }
    }
    fn from_bool(array: &Array<bool>) -> Array<Df32> {
        array.to_df32()
    }
    fn from_num(num: PyNum) -> PyResult<Df32> {
        Ok(Df32::from(f64::from_num(num)?))
    }
}

impl Elem for u32 {
    const DTYPE: DType = DType::U32;
    fn wrap(array: Array<u32>) -> Inner {
        Inner::U32(array)
    }
    fn take(inner: Inner) -> Option<Array<u32>> {
        match inner {
            Inner::U32(a) => Some(a),
            _ => None,
        }
    }
    fn from_bool(array: &Array<bool>) -> Array<u32> {
        array.to_u32()
    }
    fn from_num(num: PyNum) -> PyResult<u32> {
        match num {
            PyNum::Bool(b) => Ok(u32::from(b)),
            PyNum::Int(i) => int_range(i, DType::U32),
            PyNum::Float(_) => Err(float_into_int(DType::U32)),
        }
    }
    fn scalar(s: &ScalarInner) -> Option<luma::Scalar<u32>> {
        match s {
            ScalarInner::U32(s) => Some(s.clone()),
            _ => None,
        }
    }
}

impl Elem for i32 {
    const DTYPE: DType = DType::I32;
    fn wrap(array: Array<i32>) -> Inner {
        Inner::I32(array)
    }
    fn take(inner: Inner) -> Option<Array<i32>> {
        match inner {
            Inner::I32(a) => Some(a),
            _ => None,
        }
    }
    fn from_bool(array: &Array<bool>) -> Array<i32> {
        array.to_i32()
    }
    fn from_num(num: PyNum) -> PyResult<i32> {
        match num {
            PyNum::Bool(b) => Ok(i32::from(b)),
            PyNum::Int(i) => int_range(i, DType::I32),
            PyNum::Float(_) => Err(float_into_int(DType::I32)),
        }
    }
    fn scalar(s: &ScalarInner) -> Option<luma::Scalar<i32>> {
        match s {
            ScalarInner::I32(s) => Some(s.clone()),
            _ => None,
        }
    }
}

impl Elem for bool {
    const DTYPE: DType = DType::Bool;
    fn wrap(array: Array<bool>) -> Inner {
        Inner::Bool(array)
    }
    fn take(inner: Inner) -> Option<Array<bool>> {
        match inner {
            Inner::Bool(a) => Some(a),
            _ => None,
        }
    }
    fn from_bool(array: &Array<bool>) -> Array<bool> {
        array.clone()
    }
    fn from_num(num: PyNum) -> PyResult<bool> {
        match num {
            PyNum::Bool(b) => Ok(b),
            _ => Err(PyTypeError::new_err("only a bool scalar combines with a bool array")),
        }
    }
}
