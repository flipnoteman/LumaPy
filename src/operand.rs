//! The right-hand side of a binary operation: an array, a Python scalar, a `Scalar`, or
//! anything NumPy turns into an array, with NumPy-like type promotion.

use luma::{Array, ScalarType};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use crate::array::PyArr;
use crate::dtype::{DType, Elem, PyNum};
use crate::inner::Inner;
use crate::scalar::{PyScalar, ScalarInner};

#[derive(Clone, Debug)]
pub enum Rhs {
    Arr(Inner, usize),
    Num(PyNum),
    Scalar(ScalarInner),
}

impl Rhs {
    /// Reads `obj` as an operand of an operation on `lhs`; `None` for objects that are no
    /// operand at all (Python then tries the reflected operation or raises `TypeError`).
    ///
    /// `lhs` is also the fallback when `obj` is the array being modified in place (`x += x`):
    /// it is mutably borrowed then, and a handle on it is the same data.
    pub fn extract(obj: &Bound<'_, PyAny>, lhs: &PyArr, in_place: bool) -> PyResult<Option<Rhs>> {
        if let Ok(arr) = obj.cast::<PyArr>() {
            return Ok(Some(match arr.try_borrow() {
                Ok(a) => Rhs::Arr(a.inner.clone(), a.ndim),
                Err(_) if in_place => Rhs::Arr(lhs.inner.clone(), lhs.ndim),
                Err(e) => return Err(e.into()),
            }));
        }
        if let Ok(s) = obj.cast::<PyScalar>() {
            return Ok(Some(Rhs::Scalar(s.get().inner.clone())));
        }
        if let Some(num) = PyNum::extract(obj) {
            return Ok(Some(Rhs::Num(num)));
        }
        if is_array_like(obj) {
            let ctx = lhs.inner.context();
            let arr = crate::convert::from_object(obj, Some(lhs.inner.dtype()), Some(&ctx))?;
            return Ok(Some(Rhs::Arr(arr.inner, arr.ndim)));
        }
        Ok(None)
    }

    /// The number of NumPy axes this operand has (0 for scalars).
    pub fn ndim(&self) -> usize {
        match self {
            Rhs::Arr(_, ndim) => *ndim,
            _ => 0,
        }
    }
}

/// NumPy arrays, lists and tuples, and objects with `__array__`.
pub fn is_array_like(obj: &Bound<'_, PyAny>) -> bool {
    obj.is_instance_of::<pyo3::types::PyList>()
        || obj.is_instance_of::<pyo3::types::PyTuple>()
        || obj.hasattr("__array__").unwrap_or(false)
        || obj.hasattr("__array_interface__").unwrap_or(false)
}

/// What kind of operation a pair is promoted for.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// `+ - * // % **`: booleans count as `int32`.
    Arith,
    /// `/`: integers become `float32`.
    TrueDiv,
    /// Comparisons, `minimum`/`maximum`, `where`: booleans stay booleans.
    Compare,
    /// `& | ^ << >>`: integers and booleans only.
    Bitwise,
}

/// The common type of two array dtypes, NumPy-like within Luma's types: integers and booleans
/// join a float, `bool` joins anything. `uint32` with `int32` and `float64` with `df32` have no
/// common type here and raise.
pub fn common(a: DType, b: DType) -> PyResult<DType> {
    use DType::*;
    Ok(match (a, b) {
        _ if a == b => a,
        (Bool, x) | (x, Bool) => x,
        (F32, U32 | I32) | (U32 | I32, F32) => F32,
        (F64, U32 | I32 | F32) | (U32 | I32 | F32, F64) => F64,
        (Df32, U32 | I32 | F32) | (U32 | I32 | F32, Df32) => Df32,
        _ => {
            return Err(PyTypeError::new_err(format!(
                "no common dtype for {} and {}; convert one with .astype()",
                a.name(),
                b.name()
            )));
        }
    })
}

fn target(lhs: DType, rhs: &Rhs, kind: Kind) -> PyResult<DType> {
    let mut t = match rhs {
        Rhs::Arr(inner, _) => common(lhs, inner.dtype())?,
        Rhs::Scalar(s) => common(lhs, s.dtype())?,
        Rhs::Num(PyNum::Bool(_)) => lhs,
        Rhs::Num(PyNum::Int(_)) => {
            if lhs == DType::Bool {
                DType::I32
            } else {
                lhs
            }
        }
        Rhs::Num(PyNum::Float(_)) => {
            if lhs.is_float() {
                lhs
            } else {
                DType::F32
            }
        }
    };
    match kind {
        Kind::Arith if t == DType::Bool => t = DType::I32,
        Kind::TrueDiv if !t.is_float() => t = DType::F32,
        Kind::Bitwise if t.is_float() => {
            return Err(PyTypeError::new_err(format!(
                "bitwise operations take integer or bool arrays, not {}",
                t.name()
            )));
        }
        _ => {}
    }
    Ok(t)
}

/// Converts both sides to their common type for an operation of `kind`.
pub fn promote(lhs: &Inner, rhs: Rhs, kind: Kind) -> PyResult<(Inner, Rhs)> {
    let t = target(lhs.dtype(), &rhs, kind)?;
    let lhs = lhs.astype(t)?;
    let rhs = match rhs {
        Rhs::Arr(inner, ndim) => Rhs::Arr(inner.astype(t)?, ndim),
        Rhs::Scalar(s) if s.dtype() != t => {
            return Err(PyTypeError::new_err(format!(
                "a {} Scalar cannot combine with a {} operation",
                s.dtype().name(),
                t.name()
            )));
        }
        other => other,
    };
    Ok((lhs, rhs))
}

/// The operand in the form Luma's operations take for element type `T`.
pub enum Opnd<T: ScalarType> {
    Arr(Array<T>),
    Val(T),
    Sc(luma::Scalar<T>),
}

/// `rhs` (already promoted to `T`) as an operand.
pub fn resolve<T: Elem>(rhs: Rhs, op: &str) -> PyResult<Opnd<T>> {
    match rhs {
        Rhs::Arr(inner, _) => {
            let dtype = inner.dtype();
            match inner {
                Inner::Bool(b) => Ok(Opnd::Arr(T::from_bool(&b))),
                other => T::take(other).map(Opnd::Arr).ok_or_else(|| {
                    PyTypeError::new_err(format!("{op}: a {} operand for a {} array", dtype.name(), T::DTYPE.name()))
                }),
            }
        }
        Rhs::Num(num) => Ok(Opnd::Val(T::from_num(num)?)),
        Rhs::Scalar(s) => T::scalar(&s).map(Opnd::Sc).ok_or_else(|| {
            PyTypeError::new_err(format!("{op}: a {} Scalar for a {} array", s.dtype().name(), T::DTYPE.name()))
        }),
    }
}

/// Evaluates `$e` with `$b` bound to the operand as Luma takes it:
/// * `full`: `&Array<T>`, `T` or `&Scalar<T>`;
/// * `val`: `&Array<T>` or `T` (types without `Scalar`);
/// * `arr`: `&Array<T>` only, a value becoming a one-element array that broadcasts.
#[macro_export]
macro_rules! with_opnd {
    (full, $o:expr, $ctx:expr, $b:ident => $e:expr) => {
        match $o {
            $crate::operand::Opnd::Arr(x) => {
                let $b = &x;
                $e
            }
            $crate::operand::Opnd::Val(x) => {
                let $b = x;
                $e
            }
            $crate::operand::Opnd::Sc(x) => {
                let $b = &x;
                $e
            }
        }
    };
    (val, $o:expr, $ctx:expr, $b:ident => $e:expr) => {
        match $o {
            $crate::operand::Opnd::Arr(x) => {
                let $b = &x;
                $e
            }
            $crate::operand::Opnd::Val(x) => {
                let $b = x;
                $e
            }
            $crate::operand::Opnd::Sc(_) => unreachable!("no Scalar of this type"),
        }
    };
    (arr, $o:expr, $ctx:expr, $b:ident => $e:expr) => {
        match $o {
            $crate::operand::Opnd::Arr(x) => {
                let $b = &x;
                $e
            }
            $crate::operand::Opnd::Val(x) => {
                let x = luma::Array::full_on($ctx, 1usize, x).map_err($crate::error::luma_err)?;
                let $b = &x;
                $e
            }
            $crate::operand::Opnd::Sc(_) => unreachable!("no Scalar of this type"),
        }
    };
}

/// A binary operation over the listed `Variant: type: mode` triples (see [`with_opnd`]),
/// returning `PyResult<Inner>`; other element types raise `TypeError`.
#[macro_export]
macro_rules! binop {
    ($lhs:expr, $rhs:expr, $name:expr, [$($v:ident : $t:ty : $mode:ident),+ $(,)?], $a:ident, $b:ident => $e:expr) => {{
        let rhs: $crate::operand::Rhs = $rhs;
        match $lhs {
            $(
                $crate::inner::Inner::$v($a) => {
                    let ctx = $a.context().clone();
                    let _ = &ctx;
                    let o = $crate::operand::resolve::<$t>(rhs, $name)?;
                    Ok($crate::inner::Inner::from($crate::with_opnd!($mode, o, &ctx, $b => $e)))
                }
            )+
            #[allow(unreachable_patterns)]
            other => Err($crate::inner::unsupported($name, other.dtype())),
        }
    }};
}
