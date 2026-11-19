//! The type-erased Luma array behind the Python `Array`, and the shape/axis translation between
//! NumPy order (last axis fastest) and Luma order (axis 0 fastest).

use luma::{Array, Context, Df32, Shape};
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::dtype::DType;

/// A Luma array of any supported element type. Cloning is cheap: another handle on the same
/// data or pending expression, as `luma::Array::clone`.
#[derive(Clone, Debug)]
pub enum Inner {
    F32(Array<f32>),
    F64(Array<f64>),
    Df32(Array<Df32>),
    U32(Array<u32>),
    I32(Array<i32>),
    Bool(Array<bool>),
}

macro_rules! from_array {
    ($($t:ty => $v:ident),*) => {$(
        impl From<Array<$t>> for Inner {
            fn from(a: Array<$t>) -> Inner {
                Inner::$v(a)
            }
        }
    )*};
}

from_array!(f32 => F32, f64 => F64, Df32 => Df32, u32 => U32, i32 => I32, bool => Bool);

/// Runs `$body` with `$a` bound to the typed array, for every element type.
#[macro_export]
macro_rules! each {
    ($inner:expr, $a:ident => $body:expr) => {
        match $inner {
            $crate::inner::Inner::F32($a) => $body,
            $crate::inner::Inner::F64($a) => $body,
            $crate::inner::Inner::Df32($a) => $body,
            $crate::inner::Inner::U32($a) => $body,
            $crate::inner::Inner::I32($a) => $body,
            $crate::inner::Inner::Bool($a) => $body,
        }
    };
}

/// Runs `$body` (wrapped in `Ok`) for the listed element types and raises `TypeError` for the
/// others.
#[macro_export]
macro_rules! dispatch {
    ($inner:expr, $name:expr, [$($v:ident),+ $(,)?], $a:ident => $body:expr) => {
        match $inner {
            $($crate::inner::Inner::$v($a) => Ok($body),)+
            #[allow(unreachable_patterns)]
            other => Err($crate::inner::unsupported($name, other.dtype())),
        }
    };
}

/// [`dispatch`] over every element type.
#[macro_export]
macro_rules! dispatch_scalar {
    ($inner:expr, $name:expr, $a:ident => $body:expr) => {
        $crate::dispatch!($inner, $name, [F32, F64, Df32, U32, I32, Bool], $a => $body)
    };
}

pub fn unsupported(op: &str, dtype: DType) -> PyErr {
    PyTypeError::new_err(format!("{op} is not supported for {} arrays", dtype.name()))
}

impl Inner {
    pub fn dtype(&self) -> DType {
        match self {
            Inner::F32(_) => DType::F32,
            Inner::F64(_) => DType::F64,
            Inner::Df32(_) => DType::Df32,
            Inner::U32(_) => DType::U32,
            Inner::I32(_) => DType::I32,
            Inner::Bool(_) => DType::Bool,
        }
    }

    pub fn shape(&self) -> Shape {
        each!(self, a => a.shape())
    }

    pub fn context(&self) -> Context {
        each!(self, a => a.context().clone())
    }

    pub fn len(&self) -> usize {
        each!(self, a => a.len())
    }

    pub fn reshape(&self, shape: Shape) -> Inner {
        each!(self, a => a.reshape(shape).into())
    }

    /// The array converted to `dtype` (Luma's `to_*` casts; a cast to the same type is a cheap
    /// handle). Pairs without a direct Luma cast go through `float32`.
    pub fn astype(&self, dtype: DType) -> PyResult<Inner> {
        use Inner::*;
        if self.dtype() == dtype {
            return Ok(self.clone());
        }
        Ok(match (self, dtype) {
            (F32(a), DType::U32) => a.to_u32().into(),
            (F32(a), DType::I32) => a.to_i32().into(),
            (F32(a), DType::F64) => a.to_f64().into(),
            (F32(a), DType::Df32) => a.to_df32().into(),
            (F32(a), DType::Bool) => a.ne(0.0f32).into(),
            (U32(a), DType::F32) => a.to_f32().into(),
            (U32(a), DType::I32) => a.to_i32().into(),
            (U32(a), DType::Bool) => a.ne(0u32).into(),
            (I32(a), DType::F32) => a.to_f32().into(),
            (I32(a), DType::U32) => a.to_u32().into(),
            (I32(a), DType::Bool) => a.ne(0i32).into(),
            (Bool(a), DType::U32) => a.to_u32().into(),
            (Bool(a), DType::I32) => a.to_i32().into(),
            (Bool(a), DType::F32) => a.to_f32().into(),
            (Bool(a), DType::F64) => a.to_f64().into(),
            (Bool(a), DType::Df32) => a.to_df32().into(),
            (F64(a), DType::F32) => a.to_f32().into(),
            (F64(a), DType::Df32) => a.to_df32().into(),
            (Df32(a), DType::F32) => a.to_f32().into(),
            (Df32(a), DType::F64) => a.to_f64().into(),
            // No direct cast: through float32.
            (F64(_) | Df32(_), DType::U32 | DType::I32 | DType::Bool) | (U32(_) | I32(_), DType::F64 | DType::Df32) => {
                self.astype(DType::F32)?.astype(dtype)?
            }
            _ => unreachable!("every pair of distinct dtypes is covered"),
        })
    }
}

// -- Shapes and axes ---------------------------------------------------------------------------

/// Luma arrays have at most four axes.
pub const MAX_DIMS: usize = 4;

/// A NumPy-order shape as a Luma shape (reversed, trailing axes 1). `()` is one element.
pub fn luma_shape(py_shape: &[usize]) -> PyResult<Shape> {
    if py_shape.len() > MAX_DIMS {
        return Err(PyValueError::new_err(format!(
            "Luma arrays have at most {MAX_DIMS} dimensions, got shape {py_shape:?}"
        )));
    }
    if py_shape.contains(&0) {
        return Err(PyValueError::new_err(format!("Luma arrays cannot be empty, got shape {py_shape:?}")));
    }
    let mut dims = [1usize; MAX_DIMS];
    for (k, &d) in py_shape.iter().rev().enumerate() {
        dims[k] = d;
    }
    Ok(Shape::new(dims))
}

/// The number of axes a Luma shape needs (its last extent above 1).
pub fn rank(shape: Shape) -> usize {
    let dims = shape.dims();
    (0..MAX_DIMS).rev().find(|&k| dims[k] != 1).map_or(0, |k| k + 1)
}

/// The NumPy-order shape of `ndim` axes.
pub fn py_dims(shape: Shape, ndim: usize) -> Vec<usize> {
    let dims = shape.dims();
    (0..ndim).rev().map(|k| dims[k]).collect()
}

/// A NumPy axis (negative counts from the end) as the Luma axis.
pub fn luma_axis(axis: isize, ndim: usize) -> PyResult<usize> {
    let n = ndim as isize;
    let a = if axis < 0 { axis + n } else { axis };
    if a < 0 || a >= n {
        return Err(PyIndexError::new_err(format!("axis {axis} is out of bounds for an array of dimension {ndim}")));
    }
    Ok(ndim - 1 - a as usize)
}

/// A NumPy axis normalized to `0..ndim`.
pub fn norm_axis(axis: isize, ndim: usize) -> PyResult<usize> {
    Ok(ndim - 1 - luma_axis(axis, ndim)?)
}

/// A shape argument: an int or a sequence of ints.
pub fn extract_shape(obj: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    if let Ok(n) = obj.extract::<usize>() {
        return Ok(vec![n]);
    }
    obj.extract::<Vec<usize>>().map_err(|_| PyTypeError::new_err("a shape is an int or a sequence of ints"))
}
