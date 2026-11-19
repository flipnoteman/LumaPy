//! Operators, comparisons and elementwise math on `lumapy.Array`.

use luma::{Array, Df32};
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::pyclass::CompareOp;

use crate::array::PyArr;
use crate::convert;
use crate::dtype::{DType, PyNum};
use crate::error::IntoPy;
use crate::inner::Inner;
use crate::operand::{Kind, Rhs, promote};
use crate::{binop, dispatch};

// -- The operations on promoted operands --------------------------------------------------------

pub fn add(l: &Inner, r: Rhs) -> PyResult<Inner> {
    binop!(l, r, "add", [F32: f32: full, U32: u32: full, I32: i32: full, F64: f64: val, Df32: Df32: val], a, b => a + b)
}

pub fn sub(l: &Inner, r: Rhs) -> PyResult<Inner> {
    binop!(l, r, "subtract", [F32: f32: full, U32: u32: full, I32: i32: full, F64: f64: val, Df32: Df32: val], a, b => a - b)
}

pub fn mul(l: &Inner, r: Rhs) -> PyResult<Inner> {
    binop!(l, r, "multiply", [F32: f32: full, U32: u32: full, I32: i32: full, F64: f64: val, Df32: Df32: val], a, b => a * b)
}

/// Luma's division: true division of floats, truncating division of integers.
pub fn div(l: &Inner, r: Rhs) -> PyResult<Inner> {
    binop!(l, r, "divide", [F32: f32: full, U32: u32: full, I32: i32: full, F64: f64: val, Df32: Df32: val], a, b => a / b)
}

/// Luma's remainder of integers (truncating, the sign of the dividend).
pub fn rem(l: &Inner, r: Rhs) -> PyResult<Inner> {
    binop!(l, r, "remainder", [U32: u32: full, I32: i32: full], a, b => a % b)
}

pub fn minimum(l: &Inner, r: Rhs) -> PyResult<Inner> {
    binop!(l, r, "minimum", [F32: f32: full, U32: u32: full, I32: i32: full, Df32: Df32: val, Bool: bool: val, F64: f64: arr], a, b => a.minimum(b))
}

pub fn maximum(l: &Inner, r: Rhs) -> PyResult<Inner> {
    binop!(l, r, "maximum", [F32: f32: full, U32: u32: full, I32: i32: full, Df32: Df32: val, Bool: bool: val, F64: f64: arr], a, b => a.maximum(b))
}

macro_rules! comparison {
    ($($name:ident),*) => {$(
        pub fn $name(l: &Inner, r: Rhs) -> PyResult<Inner> {
            binop!(l, r, stringify!($name), [F32: f32: full, U32: u32: full, I32: i32: full, Df32: Df32: val, Bool: bool: val, F64: f64: arr], a, b => a.$name(b))
        }
    )*};
}

comparison!(lt, le, gt, ge, eq, ne);

pub fn bitand(l: &Inner, r: Rhs) -> PyResult<Inner> {
    binop!(l, r, "bitwise_and", [U32: u32: full, I32: i32: full, Bool: bool: val], a, b => a & b)
}

pub fn bitor(l: &Inner, r: Rhs) -> PyResult<Inner> {
    binop!(l, r, "bitwise_or", [U32: u32: full, I32: i32: full, Bool: bool: val], a, b => a | b)
}

pub fn bitxor(l: &Inner, r: Rhs) -> PyResult<Inner> {
    binop!(l, r, "bitwise_xor", [U32: u32: full, I32: i32: full, Bool: bool: val], a, b => a ^ b)
}

pub fn shl(l: &Inner, r: Rhs) -> PyResult<Inner> {
    binop!(l, r, "left_shift", [U32: u32: full, I32: i32: full], a, b => a << b)
}

pub fn shr(l: &Inner, r: Rhs) -> PyResult<Inner> {
    binop!(l, r, "right_shift", [U32: u32: full, I32: i32: full], a, b => a >> b)
}

pub fn atan2(l: &Inner, r: Rhs) -> PyResult<Inner> {
    binop!(l, r, "arctan2", [F32: f32: full], a, b => a.atan2(b))
}

pub fn hypot(l: &Inner, r: Rhs) -> PyResult<Inner> {
    binop!(l, r, "hypot", [F32: f32: full], a, b => a.hypot(b))
}

pub fn copysign(l: &Inner, r: Rhs) -> PyResult<Inner> {
    binop!(l, r, "copysign", [F32: f32: full], a, b => a.copysign(b))
}

pub fn rotate_left(l: &Inner, r: Rhs) -> PyResult<Inner> {
    binop!(l, r, "rotate_left", [U32: u32: full], a, b => a.rotate_left(b))
}

pub fn rotate_right(l: &Inner, r: Rhs) -> PyResult<Inner> {
    binop!(l, r, "rotate_right", [U32: u32: full], a, b => a.rotate_right(b))
}

fn rhs_lt_zero(r: &Rhs) -> PyResult<Rhs> {
    Ok(match r {
        Rhs::Arr(inner, ndim) => Rhs::Arr(lt(inner, Rhs::Num(PyNum::Int(0)))?, *ndim),
        Rhs::Num(PyNum::Int(i)) => Rhs::Num(PyNum::Bool(*i < 0)),
        Rhs::Num(PyNum::Float(f)) => Rhs::Num(PyNum::Bool(*f < 0.0)),
        Rhs::Num(PyNum::Bool(_)) => Rhs::Num(PyNum::Bool(false)),
        Rhs::Scalar(_) => return Err(PyTypeError::new_err("// and % do not take a Scalar divisor of signed integers")),
    })
}

/// Python's `//`: rounds toward negative infinity.
pub fn floordiv(l: &Inner, r: Rhs) -> PyResult<Inner> {
    match l.dtype() {
        DType::U32 => div(l, r),
        DType::I32 => {
            // Truncating quotient, minus one where the remainder is non-zero and the signs differ.
            let q = div(l, r.clone())?;
            let rm = rem(l, r.clone())?;
            let differ = bitxor(&lt(&rm, Rhs::Num(PyNum::Int(0)))?, rhs_lt_zero(&r)?)?;
            let fix = bitand(&ne(&rm, Rhs::Num(PyNum::Int(0)))?, Rhs::Arr(differ, 0))?;
            sub(&q, Rhs::Arr(fix.astype(DType::I32)?, 0))
        }
        DType::F32 | DType::F64 => unary(&div(l, r)?, "floor"),
        other => Err(crate::inner::unsupported("floor_divide", other)),
    }
}

/// Python's `%`: the result has the sign of the divisor.
pub fn modulo(l: &Inner, r: Rhs) -> PyResult<Inner> {
    match l.dtype() {
        DType::U32 => rem(l, r),
        DType::I32 => {
            let rm = rem(l, r.clone())?;
            let differ = bitxor(&lt(&rm, Rhs::Num(PyNum::Int(0)))?, rhs_lt_zero(&r)?)?;
            let fix = bitand(&ne(&rm, Rhs::Num(PyNum::Int(0)))?, Rhs::Arr(differ, 0))?;
            // r + where(fix, divisor, 0)
            let divisor = match &r {
                Rhs::Arr(inner, _) => inner.clone(),
                Rhs::Num(n) => convert::full(&l.context(), 1usize.into(), DType::I32, *n)?,
                Rhs::Scalar(_) => unreachable!("rejected by rhs_lt_zero"),
            };
            let add_back = mul(&fix.astype(DType::I32)?, Rhs::Arr(divisor, 0))?;
            add(&rm, Rhs::Arr(add_back, 0))
        }
        DType::F32 | DType::F64 => {
            let q = floordiv(l, r.clone())?;
            let prod = match r {
                Rhs::Arr(inner, ndim) => mul(&q, Rhs::Arr(inner, ndim))?,
                other => mul(&q, other)?,
            };
            sub(l, Rhs::Arr(prod, 0))
        }
        other => Err(crate::inner::unsupported("remainder", other)),
    }
}

/// `base ** e` for an integer `e` by square-and-multiply (exact for integers, defined for
/// negative float bases, and the products fuse into one kernel).
fn int_power(base: &Inner, e: i64) -> PyResult<Inner> {
    let mut n = e.unsigned_abs();
    let mut base = base.clone();
    let mut acc: Option<Inner> = None;
    while n > 0 {
        if n & 1 == 1 {
            acc = Some(match acc {
                None => base.clone(),
                Some(acc) => mul(&acc, Rhs::Arr(base.clone(), 0))?,
            });
        }
        n >>= 1;
        if n > 0 {
            base = mul(&base, Rhs::Arr(base.clone(), 0))?;
        }
    }
    let acc = match acc {
        Some(acc) => acc,
        None => convert::full(&base.context(), base.shape(), base.dtype(), PyNum::Int(1))?,
    };
    if e < 0 {
        let one = convert::full(&acc.context(), 1usize.into(), acc.dtype(), PyNum::Int(1))?;
        return div(&one, Rhs::Arr(acc, 0));
    }
    Ok(acc)
}

/// Exponents up to this magnitude that are integers are multiplied out instead of `powf`.
const MAX_MULTIPLIED_EXPONENT: f64 = 64.0;

/// `l ** r`.
pub fn power(l: &Inner, r: Rhs) -> PyResult<Inner> {
    let int_exponent = match r {
        Rhs::Num(PyNum::Int(e)) => Some(e),
        Rhs::Num(PyNum::Float(f)) if f.fract() == 0.0 && f.abs() <= MAX_MULTIPLIED_EXPONENT => Some(f as i64),
        _ => None,
    };
    if let Some(e) = int_exponent {
        if l.dtype().is_float() && e.unsigned_abs() as f64 <= MAX_MULTIPLIED_EXPONENT {
            return int_power(l, e);
        }
        if matches!(l.dtype(), DType::U32 | DType::I32) && e >= 0 {
            return int_power(l, e);
        }
    }
    match (l, &r) {
        (Inner::F32(a), Rhs::Num(n)) => Ok(a.powf(<f32 as crate::dtype::Elem>::from_num(*n)?).into()),
        (Inner::Df32(a), Rhs::Num(n)) => Ok(a.powf(<Df32 as crate::dtype::Elem>::from_num(*n)?).into()),
        (Inner::F32(a), Rhs::Arr(Inner::F32(b), _)) => Ok((b * &a.ln()).exp().into()),
        (Inner::F64(_), _) => {
            Err(PyTypeError::new_err("** is not supported for float64 arrays (Luma has no float64 exp/ln)"))
        }
        _ => Err(PyTypeError::new_err(format!("unsupported operands for **: {} array and {:?}", l.dtype().name(), r))),
    }
}

/// The elementwise function `name` (Luma's method names).
pub fn unary(inner: &Inner, name: &str) -> PyResult<Inner> {
    macro_rules! table {
        ($($n:ident: [$($v:ident),*]);* $(;)?) => {
            match name {
                $(stringify!($n) => {
                    let supported = [$(DType::$v),*];
                    let input = if !supported.contains(&inner.dtype()) && !inner.dtype().is_float() && supported.contains(&DType::F32) {
                        inner.astype(DType::F32)?
                    } else {
                        inner.clone()
                    };
                    dispatch!(&input, stringify!($n), [$($v),*], a => Inner::from(a.$n()))
                })*
                other => Err(PyValueError::new_err(format!("unknown elementwise function {other}"))),
            }
        };
    }
    table! {
        sqrt: [F32, F64, Df32];
        exp: [F32, Df32];
        ln: [F32, Df32];
        sin: [F32];
        cos: [F32];
        tan: [F32];
        tanh: [F32, Df32];
        log1p: [F32, Df32];
        expm1: [F32, Df32];
        floor: [F32, F64];
        ceil: [F32, F64];
        trunc: [F32, F64];
        round: [F32, F64];
        abs: [F32, F64, Df32, I32];
        sign: [F32, F64, I32];
        is_finite: [F32, F64, Df32];
        is_nan: [F32, F64, Df32];
        is_inf: [F32, F64, Df32];
    }
}

/// Rounding of integers is the identity; `abs` of unsigned too.
fn unary_int_identity(inner: &Inner, name: &str) -> Option<PyResult<Inner>> {
    let int = matches!(inner.dtype(), DType::U32 | DType::I32 | DType::Bool);
    match name {
        "floor" | "ceil" | "trunc" | "round" if int => Some(Ok(inner.clone())),
        "abs" if matches!(inner.dtype(), DType::U32 | DType::Bool) => Some(Ok(inner.clone())),
        "sign" if inner.dtype() == DType::U32 => {
            Some(ne(inner, Rhs::Num(PyNum::Int(0))).and_then(|b| b.astype(DType::U32)))
        }
        "is_finite" if int => Some(eq(inner, Rhs::Arr(inner.clone(), 0))),
        "is_nan" | "is_inf" if int => Some(ne(inner, Rhs::Arr(inner.clone(), 0))),
        _ => None,
    }
}

pub fn neg(inner: &Inner) -> PyResult<Inner> {
    dispatch!(inner, "negative", [F32, F64, Df32, I32], a => Inner::from(-a))
}

pub fn invert(inner: &Inner) -> PyResult<Inner> {
    dispatch!(inner, "invert", [U32, I32, Bool], a => Inner::from(!a))
}

/// Overwrites `target` in place with `value` of the same dtype (every handle on the data sees it).
pub fn assign(target: &mut Inner, value: Inner) -> PyResult<()> {
    let dtype = target.dtype();
    match (target, value) {
        (Inner::F32(t), Inner::F32(v)) => t.assign(&v),
        (Inner::F64(t), Inner::F64(v)) => t.assign(&v),
        (Inner::Df32(t), Inner::Df32(v)) => t.assign(&v),
        (Inner::U32(t), Inner::U32(v)) => t.assign(&v),
        (Inner::I32(t), Inner::I32(v)) => t.assign(&v),
        (Inner::Bool(t), Inner::Bool(v)) => t.assign(&v),
        (_, v) => {
            return Err(PyTypeError::new_err(format!(
                "cannot store {} values in place into a {} array",
                v.dtype().name(),
                dtype.name()
            )));
        }
    }
    .py()
}

type BinFn = fn(&Inner, Rhs) -> PyResult<Inner>;

impl PyArr {
    /// `self op other`; `None` when `other` is no operand (NotImplemented).
    fn binary(&self, other: &Bound<'_, PyAny>, kind: Kind, f: BinFn) -> PyResult<Option<PyArr>> {
        let Some(rhs) = Rhs::extract(other, self, false)? else {
            return Ok(None);
        };
        let ndim = self.ndim.max(rhs.ndim());
        let (l, r) = promote(&self.inner, rhs, kind)?;
        Ok(Some(PyArr::new(f(&l, r)?, ndim)))
    }

    /// `other op self` for a non-array `other`.
    fn reflected(&self, other: &Bound<'_, PyAny>, kind: Kind, f: BinFn) -> PyResult<Option<PyArr>> {
        let Some(rhs) = Rhs::extract(other, self, false)? else {
            return Ok(None);
        };
        let ndim = self.ndim.max(rhs.ndim());
        let ctx = self.inner.context();
        let lhs = match rhs {
            Rhs::Arr(inner, _) => inner,
            Rhs::Num(n) => {
                let dtype = match (n, self.inner.dtype()) {
                    (PyNum::Float(_), d) if !d.is_float() => DType::F32,
                    (PyNum::Int(_), DType::Bool) => DType::I32,
                    (_, d) => d,
                };
                convert::full(&ctx, 1usize.into(), dtype, n)?
            }
            Rhs::Scalar(s) => {
                // A lazy one-element array that reads the scalar.
                let zero = convert::full(&ctx, 1usize.into(), s.dtype(), PyNum::Int(0))?;
                add(&zero, Rhs::Scalar(s))?
            }
        };
        let (l, r) = promote(&lhs, Rhs::Arr(self.inner.clone(), self.ndim), kind)?;
        Ok(Some(PyArr::new(f(&l, r)?, ndim)))
    }

    /// `self op= other`: computed, then written over this array's data in place.
    fn in_place(&mut self, other: &Bound<'_, PyAny>, kind: Kind, f: BinFn) -> PyResult<()> {
        let Some(rhs) = Rhs::extract(other, self, true)? else {
            return Err(PyTypeError::new_err(format!(
                "unsupported operand type for an in-place operation: {}",
                other.get_type().name()?
            )));
        };
        let (l, r) = promote(&self.inner, rhs, kind)?;
        let result = f(&l, r)?;
        if result.dtype() != self.inner.dtype() {
            return Err(PyTypeError::new_err(format!(
                "the result ({}) cannot be stored in place in a {} array",
                result.dtype().name(),
                self.inner.dtype().name()
            )));
        }
        assign(&mut self.inner, result)
    }

    pub fn apply(&self, name: &str) -> PyResult<PyArr> {
        let inner = match unary_int_identity(&self.inner, name) {
            Some(r) => r?,
            None => unary(&self.inner, name)?,
        };
        Ok(self.same(inner))
    }

    fn binary_method(&self, other: &Bound<'_, PyAny>, kind: Kind, f: BinFn, name: &str) -> PyResult<PyArr> {
        self.binary(other, kind, f)?.ok_or_else(|| {
            PyTypeError::new_err(format!(
                "{name}: unsupported operand {}",
                other.get_type().name().map(|n| n.to_string()).unwrap_or_default()
            ))
        })
    }
}

fn or_not_implemented(py: Python<'_>, r: Option<PyArr>) -> PyResult<Py<PyAny>> {
    match r {
        Some(a) => a.into_py_any(py),
        None => Ok(py.NotImplemented()),
    }
}

macro_rules! operators {
    ($($op:ident / $rop:ident / $iop:ident => $kind:ident, $f:path;)*) => {
        #[pymethods]
        impl PyArr {
            $(
                fn $op(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                    or_not_implemented(py, self.binary(other, Kind::$kind, $f)?)
                }

                fn $rop(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                    or_not_implemented(py, self.reflected(other, Kind::$kind, $f)?)
                }

                fn $iop(&mut self, other: &Bound<'_, PyAny>) -> PyResult<()> {
                    self.in_place(other, Kind::$kind, $f)
                }
            )*
        }
    };
}

operators! {
    __add__ / __radd__ / __iadd__ => Arith, add;
    __sub__ / __rsub__ / __isub__ => Arith, sub;
    __mul__ / __rmul__ / __imul__ => Arith, mul;
    __truediv__ / __rtruediv__ / __itruediv__ => TrueDiv, div;
    __floordiv__ / __rfloordiv__ / __ifloordiv__ => Arith, floordiv;
    __mod__ / __rmod__ / __imod__ => Arith, modulo;
    __and__ / __rand__ / __iand__ => Bitwise, bitand;
    __or__ / __ror__ / __ior__ => Bitwise, bitor;
    __xor__ / __rxor__ / __ixor__ => Bitwise, bitxor;
    __lshift__ / __rlshift__ / __ilshift__ => Bitwise, shl;
    __rshift__ / __rrshift__ / __irshift__ => Bitwise, shr;
}

macro_rules! unary_methods {
    ($($(#[$doc:meta])* $name:ident => $luma:literal;)*) => {
        #[pymethods]
        impl PyArr {
            $(
                $(#[$doc])*
                fn $name(&self) -> PyResult<PyArr> {
                    self.apply($luma)
                }
            )*
        }
    };
}

unary_methods! {
    /// Square root (integers are converted to float32 first, as for the other float functions).
    sqrt => "sqrt";
    exp => "exp";
    /// Natural logarithm (Luma's `ln`).
    ln => "ln";
    /// Natural logarithm, NumPy's name for `ln`.
    log => "ln";
    log1p => "log1p";
    expm1 => "expm1";
    sin => "sin";
    cos => "cos";
    tan => "tan";
    tanh => "tanh";
    floor => "floor";
    ceil => "ceil";
    trunc => "trunc";
    /// Rounds half to even, like NumPy.
    round => "round";
    abs => "abs";
    sign => "sign";
    is_finite => "is_finite";
    isfinite => "is_finite";
    is_nan => "is_nan";
    isnan => "is_nan";
    is_inf => "is_inf";
    isinf => "is_inf";
}

#[pymethods]
impl PyArr {
    fn __pow__(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        modulo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        if modulo.is_some_and(|m| !m.is_none()) {
            return Err(PyTypeError::new_err("pow() with a modulus is not supported"));
        }
        let Some(rhs) = Rhs::extract(other, self, false)? else {
            return Ok(py.NotImplemented());
        };
        let ndim = self.ndim.max(rhs.ndim());
        let base = match (&rhs, self.inner.dtype()) {
            (Rhs::Num(PyNum::Float(_)), d) | (Rhs::Arr(_, _), d) if !d.is_float() => self.inner.astype(DType::F32)?,
            _ => self.inner.clone(),
        };
        let rhs = match rhs {
            Rhs::Arr(inner, n) if base.dtype() == DType::F32 => Rhs::Arr(inner.astype(DType::F32)?, n),
            other => other,
        };
        PyArr::new(power(&base, rhs)?, ndim).into_py_any(py)
    }

    fn __rpow__(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        _modulo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        // `base ** self` with a scalar or array-like base: as a float32 array, exp(self * ln(base)).
        or_not_implemented(py, self.as_f32()?.reflected(other, Kind::TrueDiv, power)?)
    }

    fn __ipow__(
        &mut self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        _modulo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let result = self.__pow__(py, other, None)?;
        let result = result
            .bind(py)
            .cast::<PyArr>()
            .map_err(|_| PyTypeError::new_err("unsupported operand for **="))?
            .borrow()
            .inner
            .clone();
        if result.dtype() != self.inner.dtype() {
            return Err(PyTypeError::new_err("the result of **= has another dtype than the array"));
        }
        assign(&mut self.inner, result)
    }

    fn __neg__(&self) -> PyResult<PyArr> {
        match self.inner.dtype() {
            DType::Bool => Err(PyTypeError::new_err("negation of a bool array; use ~ for logical not")),
            DType::U32 => Err(PyTypeError::new_err("negation of a uint32 array")),
            _ => Ok(self.same(neg(&self.inner)?)),
        }
    }

    fn __pos__(&self) -> PyArr {
        self.same(self.inner.clone())
    }

    fn __abs__(&self) -> PyResult<PyArr> {
        self.apply("abs")
    }

    fn __invert__(&self) -> PyResult<PyArr> {
        Ok(self.same(invert(&self.inner)?))
    }

    fn __richcmp__(&self, py: Python<'_>, other: &Bound<'_, PyAny>, op: CompareOp) -> PyResult<Py<PyAny>> {
        let f: BinFn = match op {
            CompareOp::Lt => lt,
            CompareOp::Le => le,
            CompareOp::Gt => gt,
            CompareOp::Ge => ge,
            CompareOp::Eq => eq,
            CompareOp::Ne => ne,
        };
        or_not_implemented(py, self.binary(other, Kind::Compare, f)?)
    }

    /// The elementwise minimum with `other` (an array or a scalar), broadcasting.
    fn minimum(&self, other: &Bound<'_, PyAny>) -> PyResult<PyArr> {
        self.binary_method(other, Kind::Compare, minimum, "minimum")
    }

    /// The elementwise maximum with `other` (an array or a scalar), broadcasting.
    fn maximum(&self, other: &Bound<'_, PyAny>) -> PyResult<PyArr> {
        self.binary_method(other, Kind::Compare, maximum, "maximum")
    }

    /// `atan2(self, x)` elementwise (`self` is y).
    fn atan2(&self, x: &Bound<'_, PyAny>) -> PyResult<PyArr> {
        self.as_f32()?.binary_method(x, Kind::TrueDiv, atan2, "atan2")
    }

    /// `sqrt(self² + other²)` without overflow.
    fn hypot(&self, other: &Bound<'_, PyAny>) -> PyResult<PyArr> {
        self.as_f32()?.binary_method(other, Kind::TrueDiv, hypot, "hypot")
    }

    /// The magnitude of `self` with the sign of `sign`.
    fn copysign(&self, sign: &Bound<'_, PyAny>) -> PyResult<PyArr> {
        self.as_f32()?.binary_method(sign, Kind::TrueDiv, copysign, "copysign")
    }

    /// `self ** exponent` for a scalar exponent (Luma's `powf`).
    fn powf(&self, py: Python<'_>, exponent: f64) -> PyResult<Py<PyAny>> {
        self.__pow__(py, &exponent.into_pyobject(py)?.into_any(), None)
    }

    /// Luma's truncating remainder (the sign of the dividend), unlike `%`.
    fn rem(&self, other: &Bound<'_, PyAny>) -> PyResult<PyArr> {
        self.binary_method(other, Kind::Arith, rem, "rem")
    }

    /// The values limited to `[lo, hi]`.
    #[pyo3(signature = (lo = None, hi = None))]
    fn clamp(&self, py: Python<'_>, lo: Option<f64>, hi: Option<f64>) -> PyResult<PyArr> {
        if let (Inner::F32(a), Some(lo), Some(hi)) = (&self.inner, lo, hi) {
            return Ok(self.same(a.clamp(lo as f32, hi as f32).into()));
        }
        let mut out = self.same(self.inner.clone());
        if let Some(lo) = lo {
            out = out.maximum(&lo.into_pyobject(py)?.into_any())?;
        }
        if let Some(hi) = hi {
            out = out.minimum(&hi.into_pyobject(py)?.into_any())?;
        }
        Ok(out)
    }

    /// NumPy's name for `clamp`.
    #[pyo3(signature = (min = None, max = None))]
    fn clip(&self, py: Python<'_>, min: Option<f64>, max: Option<f64>) -> PyResult<PyArr> {
        self.clamp(py, min, max)
    }

    /// Shifts left by `amount` bits (integers).
    fn shl(&self, amount: &Bound<'_, PyAny>) -> PyResult<PyArr> {
        self.binary_method(amount, Kind::Bitwise, shl, "shl")
    }

    /// Shifts right by `amount` bits (logical for uint32, arithmetic for int32).
    fn shr(&self, amount: &Bound<'_, PyAny>) -> PyResult<PyArr> {
        self.binary_method(amount, Kind::Bitwise, shr, "shr")
    }

    /// Rotates the bits of uint32 values left.
    fn rotate_left(&self, amount: &Bound<'_, PyAny>) -> PyResult<PyArr> {
        self.binary_method(amount, Kind::Bitwise, rotate_left, "rotate_left")
    }

    /// Rotates the bits of uint32 values right.
    fn rotate_right(&self, amount: &Bound<'_, PyAny>) -> PyResult<PyArr> {
        self.binary_method(amount, Kind::Bitwise, rotate_right, "rotate_right")
    }

    /// The matrix product (`float32`; other types are converted). 1-D operands are vectors, as
    /// in NumPy.
    fn __matmul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let Some(rhs) = Rhs::extract(other, self, false)? else {
            return Ok(py.NotImplemented());
        };
        let Rhs::Arr(inner, ndim) = rhs else {
            return Err(PyValueError::new_err("matmul: an operand is a scalar; use *"));
        };
        matmul(self, &PyArr::new(inner, ndim))?.into_py_any(py)
    }

    fn __rmatmul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let Some(Rhs::Arr(inner, ndim)) = Rhs::extract(other, self, false)? else {
            return Ok(py.NotImplemented());
        };
        matmul(&PyArr::new(inner, ndim), self)?.into_py_any(py)
    }

    /// The matrix product `self @ other`.
    fn matmul(&self, other: PyRef<'_, PyArr>) -> PyResult<PyArr> {
        matmul(self, &other)
    }
}

impl PyArr {
    pub fn as_f32(&self) -> PyResult<PyArr> {
        Ok(self.same(self.inner.astype(DType::F32)?))
    }
}

fn f32_operand(a: &PyArr) -> PyResult<Array<f32>> {
    match a.inner.dtype() {
        DType::F64 | DType::Df32 => {
            Err(PyTypeError::new_err("matmul is float32 only; convert with .astype(\"float32\")"))
        }
        _ => match a.inner.astype(DType::F32)? {
            Inner::F32(x) => Ok(x),
            _ => unreachable!(),
        },
    }
}

/// NumPy's `matmul` for 1-D and 2-D operands, on Luma's `[K, M] x [N, K] -> [N, M]` product.
pub fn matmul(a: &PyArr, b: &PyArr) -> PyResult<PyArr> {
    if a.ndim > 2 || b.ndim > 2 {
        return Err(PyValueError::new_err("matmul takes 1-D or 2-D arrays (no batched products)"));
    }
    let x = f32_operand(a)?;
    let mut y = f32_operand(b)?;
    if b.ndim <= 1 {
        // A vector on the right is a column: NumPy (K, 1), Luma [1, K].
        y = y.reshape([1, y.len()]);
    }
    let product = x.matmul(&y);
    let (m, n) = (product.shape().dims()[1], product.shape().dims()[0]);
    let shape: Vec<usize> = match (a.ndim <= 1, b.ndim <= 1) {
        (false, false) => vec![m, n],
        (true, false) => vec![n],
        (false, true) => vec![m],
        (true, true) => vec![],
    };
    PyArr::new(Inner::F32(product), 0).with_shape(&shape)
}
