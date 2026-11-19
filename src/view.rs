//! Views (reshape, transpose, slices, flips, pads, shifts, broadcasts) and indexing.

use std::ops::Range;

use luma::{Array, Df32, PadMode};
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyEllipsis, PySlice, PySliceMethods, PyTuple};

use crate::array::PyArr;
use crate::convert;
use crate::dispatch_scalar;
use crate::dtype::{DType, Elem, PyNum};
use crate::error::IntoPy;
use crate::inner::{Inner, MAX_DIMS, extract_shape, luma_axis, luma_shape, norm_axis};
use crate::ops;
use crate::reduce::Mask;

/// One step of a basic index: a range along a Luma axis, read backwards when `flip`.
#[derive(Clone, Debug)]
struct Step {
    axis: usize,
    range: Range<usize>,
    flip: bool,
}

/// A parsed basic index: the steps, and the NumPy shape of the result.
struct Basic {
    steps: Vec<Step>,
    shape: Vec<usize>,
}

fn parse_basic(key: &Bound<'_, PyAny>, arr: &PyArr) -> PyResult<Basic> {
    let items: Vec<Bound<'_, PyAny>> = match key.cast::<PyTuple>() {
        Ok(t) => t.iter().collect(),
        Err(_) => vec![key.clone()],
    };
    let ellipses = items.iter().filter(|i| i.is_instance_of::<PyEllipsis>()).count();
    if ellipses > 1 {
        return Err(PyIndexError::new_err("an index can only have a single ellipsis"));
    }
    let consuming = items.iter().filter(|i| !i.is_none() && !i.is_instance_of::<PyEllipsis>()).count();
    if consuming > arr.ndim {
        return Err(PyIndexError::new_err(format!(
            "too many indices: the array is {}-dimensional, {consuming} were indexed",
            arr.ndim
        )));
    }
    let dims = arr.py_shape();
    let mut steps = Vec::new();
    let mut shape = Vec::new();
    let mut axis = 0usize; // the NumPy axis of the source the next item consumes
    for item in &items {
        if item.is_instance_of::<PyEllipsis>() {
            for _ in 0..arr.ndim - consuming {
                shape.push(dims[axis]);
                axis += 1;
            }
            continue;
        }
        if item.is_none() {
            shape.push(1);
            continue;
        }
        let extent = dims[axis];
        let la = arr.ndim - 1 - axis;
        if let Ok(slice) = item.cast::<PySlice>() {
            let ind = slice.indices(extent as isize)?;
            let (start, stop, step) = (ind.start, ind.stop, ind.step);
            let (range, flip) = match step {
                1 => (start.max(0) as usize..stop.max(start).max(0) as usize, false),
                -1 => ((stop + 1).max(0) as usize..(start + 1).max(stop + 1).max(0) as usize, true),
                _ => {
                    return Err(PyIndexError::new_err(
                        "slices take a step of 1 or -1 (use take() for strided gathers)",
                    ));
                }
            };
            if range.is_empty() {
                return Err(PyIndexError::new_err("the slice selects no elements (Luma has no empty arrays)"));
            }
            shape.push(range.len());
            if range != (0..extent) || flip {
                steps.push(Step { axis: la, range, flip });
            }
        } else if let Ok(i) = item.extract::<isize>() {
            let k = if i < 0 { i + extent as isize } else { i };
            if k < 0 || k >= extent as isize {
                return Err(PyIndexError::new_err(format!(
                    "index {i} is out of bounds for axis {axis} with size {extent}"
                )));
            }
            steps.push(Step { axis: la, range: k as usize..k as usize + 1, flip: false });
        } else {
            return Err(PyIndexError::new_err(
                "indices are ints, slices, None, ..., a bool mask array or (for 1-D arrays) an integer array",
            ));
        }
        axis += 1;
    }
    while axis < arr.ndim {
        shape.push(dims[axis]);
        axis += 1;
    }
    Ok(Basic { steps, shape })
}

/// What an index key is beyond basic indexing.
enum Advanced {
    Mask(Mask),
    Take(Array<u32>, usize),
}

fn parse_advanced(key: &Bound<'_, PyAny>, arr: &PyArr) -> PyResult<Option<Advanced>> {
    let candidate = if let Ok(a) = key.cast::<PyArr>() {
        let a = a.try_borrow()?;
        Some(PyArr::new(a.inner.clone(), a.ndim))
    } else if key.is_instance_of::<pyo3::types::PyList>() || key.hasattr("__array__")? {
        Some(convert::from_object(key, None, Some(&arr.inner.context()))?)
    } else {
        None
    };
    let Some(idx) = candidate else {
        return Ok(None);
    };
    match idx.inner.dtype() {
        DType::Bool => Ok(Some(Advanced::Mask(Mask::from_arr(idx.inner)?))),
        DType::U32 | DType::I32 => {
            if arr.ndim > 1 {
                return Err(PyIndexError::new_err(
                    "integer-array indexing is supported on 1-D arrays; use take() for flat gathers",
                ));
            }
            let ndim = idx.ndim;
            match idx.inner.astype(DType::U32)? {
                Inner::U32(u) => Ok(Some(Advanced::Take(u, ndim))),
                _ => unreachable!(),
            }
        }
        _ => Err(PyIndexError::new_err("index arrays are integer or bool arrays")),
    }
}

fn apply_steps(inner: &Inner, steps: &[Step]) -> PyResult<Inner> {
    let mut cur = inner.clone();
    for s in steps {
        cur = dispatch_scalar!(&cur, "slice", a => Inner::from(a.slice(s.axis, s.range.clone())))?;
        if s.flip {
            cur = dispatch_scalar!(&cur, "flip", a => Inner::from(a.flip(s.axis)))?;
        }
    }
    Ok(cur)
}

/// The value of an assignment, as an array of `T` or a value.
enum Value {
    Arr(Inner),
    Num(PyNum),
}

fn value_of(obj: &Bound<'_, PyAny>, target: &PyArr) -> PyResult<Value> {
    if let Ok(a) = obj.cast::<PyArr>() {
        let inner = match a.try_borrow() {
            Ok(a) => a.inner.clone(),
            Err(_) => target.inner.clone(), // `x[...] = x`
        };
        return Ok(Value::Arr(inner.astype(target.inner.dtype())?));
    }
    if let Some(n) = PyNum::extract(obj) {
        return Ok(Value::Num(n));
    }
    Ok(Value::Arr(convert::from_object(obj, Some(target.inner.dtype()), Some(&target.inner.context()))?.inner))
}

/// Writes `value` through the view `steps` describe (Luma's `slice_mut`/`flip_mut` chain).
macro_rules! write_view {
    ($a:expr, $t:ty, $steps:expr, $value:expr, $num:ident, $shape:ident => $num_e:expr) => {{
        let steps: &[Step] = $steps;
        let first = &steps[0];
        let mut view = $a.slice_mut(first.axis, first.range.clone());
        if first.flip {
            view = view.flip(first.axis);
        }
        for s in &steps[1..] {
            view = view.slice(s.axis, s.range.clone());
            if s.flip {
                view = view.flip(s.axis);
            }
        }
        let $shape = view.shape();
        let shape = $shape;
        match $value {
            Value::Arr(inner) => {
                let inner =
                    if inner.len() == shape.len() && inner.shape() != shape { inner.reshape(shape) } else { inner };
                let v: Array<$t> = <$t>::take(inner).expect("converted to the target dtype");
                view.assign(&v).py()
            }
            Value::Num(n) => {
                let $num = <$t>::from_num(n)?;
                view.assign($num_e).py()
            }
        }
    }};
}

fn write_steps(target: &mut Inner, steps: &[Step], value: Value) -> PyResult<()> {
    match target {
        Inner::F32(a) => write_view!(a, f32, steps, value, v, _shape => v),
        Inner::U32(a) => write_view!(a, u32, steps, value, v, _shape => v),
        Inner::I32(a) => write_view!(a, i32, steps, value, v, _shape => v),
        Inner::Bool(a) => write_view!(a, bool, steps, value, v, _shape => v),
        Inner::Df32(a) => write_view!(a, Df32, steps, value, v, _shape => v),
        Inner::F64(a) => {
            let ctx = a.context().clone();
            write_view!(a, f64, steps, value, v, shape => &Array::full_on(&ctx, shape, v).py()?)
        }
    }
}

fn pad_mode(mode: &str) -> PyResult<PadMode> {
    Ok(match mode {
        "constant" | "zero" => PadMode::Zero,
        "edge" => PadMode::Edge,
        "wrap" => PadMode::Wrap,
        "symmetric" | "mirror" => PadMode::Mirror,
        other => {
            return Err(PyValueError::new_err(format!(
                "pad mode is \"constant\", \"edge\", \"wrap\" or \"symmetric\", not {other:?}"
            )));
        }
    })
}

/// NumPy's `pad_width`: an int, a `(before, after)` pair, or one pair per axis.
fn pad_widths(obj: &Bound<'_, PyAny>, ndim: usize) -> PyResult<Vec<(usize, usize)>> {
    if let Ok(w) = obj.extract::<usize>() {
        return Ok(vec![(w, w); ndim]);
    }
    // A pair of ints is one (before, after) for every axis, as in NumPy.
    if let Ok(pair) = obj.extract::<(usize, usize)>() {
        return Ok(vec![pair; ndim]);
    }
    if let Ok(list) = obj.extract::<Vec<(usize, usize)>>() {
        if list.len() == ndim {
            return Ok(list);
        }
        if list.len() == 1 {
            return Ok(vec![list[0]; ndim]);
        }
    }
    if let Ok(list) = obj.extract::<Vec<usize>>()
        && list.len() == ndim
    {
        return Ok(list.into_iter().map(|w| (w, w)).collect());
    }
    Err(PyValueError::new_err(format!("pad_width is an int, a (before, after) pair, or {ndim} of them")))
}

/// Per-axis offsets (NumPy order, `axis` as in `np.roll`) as Luma offsets.
fn offsets(shift: &Bound<'_, PyAny>, axis: Option<&Bound<'_, PyAny>>, ndim: usize) -> PyResult<[isize; 4]> {
    let mut out = [0isize; MAX_DIMS];
    let shifts: Vec<isize> = match shift.extract::<isize>() {
        Ok(s) => vec![s],
        Err(_) => shift.extract()?,
    };
    let axes: Vec<isize> = match axis.filter(|a| !a.is_none()) {
        Some(a) => match a.extract::<isize>() {
            Ok(a) => vec![a],
            Err(_) => a.extract()?,
        },
        None if shifts.len() == ndim => (0..ndim as isize).collect(),
        None if ndim <= 1 => vec![0],
        None => {
            return Err(PyValueError::new_err(format!(
                "give one offset per axis ({ndim}) or name the axes with axis="
            )));
        }
    };
    let shifts = if shifts.len() == 1 && axes.len() > 1 { vec![shifts[0]; axes.len()] } else { shifts };
    if shifts.len() != axes.len() {
        return Err(PyValueError::new_err("as many offsets as axes"));
    }
    for (s, a) in shifts.into_iter().zip(axes) {
        out[luma_axis(a, ndim.max(1))?] += s;
    }
    Ok(out)
}

#[pymethods]
impl PyArr {
    /// The array with a new NumPy shape of the same size (`-1` infers one extent). Lazy.
    #[pyo3(signature = (*shape))]
    fn reshape(&self, shape: &Bound<'_, PyTuple>) -> PyResult<PyArr> {
        let dims: Vec<isize> = if shape.len() == 1 {
            let item = shape.get_item(0)?;
            match item.extract::<isize>() {
                Ok(d) => vec![d],
                Err(_) => item.extract()?,
            }
        } else {
            shape.extract()?
        };
        let size = self.inner.len();
        let known: usize = dims.iter().filter(|&&d| d >= 0).map(|&d| d as usize).product();
        let inferred = dims.iter().filter(|&&d| d < 0).count();
        if inferred > 1 {
            return Err(PyValueError::new_err("only one extent can be -1"));
        }
        let dims: Vec<usize> =
            dims.iter().map(|&d| if d < 0 { size.checked_div(known).unwrap_or(0) } else { d as usize }).collect();
        if dims.iter().product::<usize>() != size {
            return Err(PyValueError::new_err(format!("cannot reshape an array of size {size} into shape {dims:?}")));
        }
        self.with_shape(&dims)
    }

    /// The array as one axis.
    fn ravel(&self) -> PyResult<PyArr> {
        self.with_shape(&[self.inner.len()])
    }

    /// The array as one axis (a lazy view, unlike NumPy's copy; use `.copy()` for one).
    fn flatten(&self) -> PyResult<PyArr> {
        self.ravel()
    }

    /// The array with its axes permuted (reversed when no axes are given). Lazy, read-only.
    #[pyo3(signature = (*axes))]
    fn transpose(&self, axes: &Bound<'_, PyTuple>) -> PyResult<PyArr> {
        let n = self.ndim;
        let perm: Vec<isize> = if axes.is_empty() {
            (0..n as isize).rev().collect()
        } else if axes.len() == 1 && axes.get_item(0)?.extract::<isize>().is_err() {
            axes.get_item(0)?.extract()?
        } else {
            axes.extract()?
        };
        if perm.len() != n {
            return Err(PyValueError::new_err("axes don't match the array's dimensions"));
        }
        let perm: Vec<usize> = perm.into_iter().map(|p| norm_axis(p, n)).collect::<PyResult<_>>()?;
        // Result NumPy axis i is source axis perm[i]: Luma axis n-1-i reads Luma axis n-1-perm[i].
        let mut lperm = [0, 1, 2, 3];
        for (i, &p) in perm.iter().enumerate() {
            lperm[n - 1 - i] = n - 1 - p;
        }
        Ok(self.same(dispatch_scalar!(&self.inner, "transpose", a => Inner::from(a.permute_axes(lperm)))?))
    }

    /// The transpose (all axes reversed).
    #[getter(T)]
    fn t(&self) -> PyResult<PyArr> {
        let n = self.ndim;
        let mut lperm = [0, 1, 2, 3];
        for (i, slot) in lperm.iter_mut().enumerate().take(n) {
            *slot = n - 1 - i;
        }
        Ok(self.same(dispatch_scalar!(&self.inner, "transpose", a => Inner::from(a.permute_axes(lperm)))?))
    }

    /// The array with axes `a` and `b` swapped. Lazy, read-only.
    fn swapaxes(&self, a: isize, b: isize) -> PyResult<PyArr> {
        let (la, lb) = (luma_axis(a, self.ndim)?, luma_axis(b, self.ndim)?);
        Ok(self.same(dispatch_scalar!(&self.inner, "swapaxes", x => Inner::from(x.swap_axes(la, lb)))?))
    }

    /// The array reversed along `axis` (all axes when `None`). Lazy, read-only.
    #[pyo3(signature = (axis = None))]
    fn flip(&self, axis: Option<&Bound<'_, PyAny>>) -> PyResult<PyArr> {
        let axes = crate::reduce::extract_axes(axis, self.ndim)?.unwrap_or_else(|| (0..self.ndim).collect());
        let mut cur = self.inner.clone();
        for a in axes {
            let la = self.ndim - 1 - a;
            cur = dispatch_scalar!(&cur, "flip", x => Inner::from(x.flip(la)))?;
        }
        Ok(self.same(cur))
    }

    /// The array broadcast to `shape` (extent-1 axes repeat; new leading axes are added). Lazy.
    fn broadcast_to(&self, shape: &Bound<'_, PyAny>) -> PyResult<PyArr> {
        let shape = extract_shape(shape)?;
        let ls = luma_shape(&shape)?;
        Ok(PyArr::new(
            dispatch_scalar!(&self.inner, "broadcast_to", a => Inner::from(a.broadcast_to(ls)))?,
            shape.len(),
        ))
    }

    /// The array padded as `np.pad`: `pad_width` per axis, `mode` `"constant"` (zeros),
    /// `"edge"`, `"wrap"` or `"symmetric"`. Lazy.
    #[pyo3(signature = (pad_width, mode = "constant"))]
    fn pad(&self, pad_width: &Bound<'_, PyAny>, mode: &str) -> PyResult<PyArr> {
        let widths = pad_widths(pad_width, self.ndim)?;
        let mut lw = [(0usize, 0usize); MAX_DIMS];
        for (i, w) in widths.into_iter().enumerate() {
            lw[self.ndim - 1 - i] = w;
        }
        let mode = pad_mode(mode)?;
        Ok(self.same(dispatch_scalar!(&self.inner, "pad", a => Inner::from(a.pad(lw, mode)))?))
    }

    /// Circular shift (`np.roll`): element `i` moves to `i + shift` along each axis. One
    /// offset per axis, or name the axes with `axis=`. Lazy: fuses into the reading kernel.
    #[pyo3(signature = (shift, axis = None))]
    fn shift(&self, shift: &Bound<'_, PyAny>, axis: Option<&Bound<'_, PyAny>>) -> PyResult<PyArr> {
        let off = offsets(shift, axis, self.ndim)?;
        Ok(self.same(dispatch_scalar!(&self.inner, "shift", a => Inner::from(a.shift(off)))?))
    }

    /// NumPy's name for `shift`.
    #[pyo3(signature = (shift, axis = None))]
    fn roll(&self, shift: &Bound<'_, PyAny>, axis: Option<&Bound<'_, PyAny>>) -> PyResult<PyArr> {
        self.shift(shift, axis)
    }

    /// A shift that does not wrap: the cells nothing moves into read `value`.
    #[pyo3(signature = (shift, value, axis = None))]
    fn shift_fill(
        &self,
        shift: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
        axis: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyArr> {
        let off = offsets(shift, axis, self.ndim)?;
        let n = PyNum::extract(value).ok_or_else(|| PyTypeError::new_err("value is a number"))?;
        let r = match &self.inner {
            Inner::F32(a) => Inner::from(a.shift_fill(off, f32::from_num(n)?)),
            Inner::F64(a) => Inner::from(a.shift_fill(off, f64::from_num(n)?)),
            Inner::Df32(a) => Inner::from(a.shift_fill(off, Df32::from_num(n)?)),
            Inner::U32(a) => Inner::from(a.shift_fill(off, u32::from_num(n)?)),
            Inner::I32(a) => Inner::from(a.shift_fill(off, i32::from_num(n)?)),
            Inner::Bool(a) => Inner::from(a.shift_fill(off, bool::from_num(n)?)),
        };
        Ok(self.same(r))
    }

    /// A shift that does not wrap: the cells nothing moves into repeat the nearest element.
    #[pyo3(signature = (shift, axis = None))]
    fn shift_clamp(&self, shift: &Bound<'_, PyAny>, axis: Option<&Bound<'_, PyAny>>) -> PyResult<PyArr> {
        let off = offsets(shift, axis, self.ndim)?;
        Ok(self.same(dispatch_scalar!(&self.inner, "shift_clamp", a => Inner::from(a.shift_clamp(off)))?))
    }

    /// A shift that does not wrap: the cells nothing moves into mirror the array about its
    /// boundary face (finite-volume ghost cells).
    #[pyo3(signature = (shift, axis = None))]
    fn shift_mirror(&self, shift: &Bound<'_, PyAny>, axis: Option<&Bound<'_, PyAny>>) -> PyResult<PyArr> {
        let off = offsets(shift, axis, self.ndim)?;
        Ok(self.same(dispatch_scalar!(&self.inner, "shift_mirror", a => Inner::from(a.shift_mirror(off)))?))
    }

    /// Each element's index along `axis` (uint32, shaped like this array).
    fn index(&self, axis: isize) -> PyResult<PyArr> {
        let la = luma_axis(axis, self.ndim)?;
        Ok(self.same(dispatch_scalar!(&self.inner, "index", a => Inner::from(a.index(la)))?))
    }

    /// The array with a new axis of extent 1 at `axis`.
    fn expand_dims(&self, axis: isize) -> PyResult<PyArr> {
        let mut shape = self.py_shape();
        let a = norm_axis(axis, self.ndim + 1)?;
        shape.insert(a, 1);
        self.with_shape(&shape)
    }

    /// The array without its extent-1 axes (or only `axis`).
    #[pyo3(signature = (axis = None))]
    fn squeeze(&self, axis: Option<&Bound<'_, PyAny>>) -> PyResult<PyArr> {
        let shape = self.py_shape();
        let drop = crate::reduce::extract_axes(axis, self.ndim)?
            .unwrap_or_else(|| (0..self.ndim).filter(|&i| shape[i] == 1).collect());
        if drop.iter().any(|&i| shape[i] != 1) {
            return Err(PyValueError::new_err("cannot squeeze an axis of extent other than 1"));
        }
        let kept: Vec<usize> = shape.iter().enumerate().filter(|(i, _)| !drop.contains(i)).map(|(_, &d)| d).collect();
        self.with_shape(&kept)
    }

    /// Basic indexing (ints, slices of step ±1, `None`, `...`) gives lazy read-only views;
    /// a bool mask array gives the selected elements (1-D, synchronizes once); an integer
    /// array gathers from a 1-D array.
    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<PyArr> {
        if let Some(adv) = parse_advanced(key, self)? {
            return match adv {
                Advanced::Mask(mask) => {
                    let inner = &self.inner;
                    let r = py.detach(|| {
                        crate::with_mask!(&mask, m => dispatch_scalar!(inner, "filter", a => a.filter(m).map(|o| o.map(Inner::from))))
                    })?;
                    match r.py()? {
                        Some(i) => Ok(PyArr::new(i, 1)),
                        None => Err(PyValueError::new_err(
                            "the mask selects no elements (Luma has no empty arrays); use filter() to get None",
                        )),
                    }
                }
                Advanced::Take(idx, ndim) => {
                    Ok(PyArr::new(dispatch_scalar!(&self.inner, "take", a => Inner::from(a.take(&idx)))?, ndim))
                }
            };
        }
        let basic = parse_basic(key, self)?;
        let r = apply_steps(&self.inner, &basic.steps)?;
        PyArr::new(r, 0).with_shape(&basic.shape)
    }

    /// Writes through basic indexing in place (`x[1:3, :] = v`), a bool mask (`x[x < 0] = 0`)
    /// or an integer array on 1-D arrays (`x[idx] = v`, a scatter).
    fn __setitem__(&mut self, key: &Bound<'_, PyAny>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        if let Some(adv) = parse_advanced(key, self)? {
            return match adv {
                Advanced::Mask(mask) => {
                    let v = match value_of(value, self)? {
                        Value::Arr(i) => crate::operand::Rhs::Arr(i, 0),
                        Value::Num(n) => crate::operand::Rhs::Num(n),
                    };
                    let current = crate::operand::Rhs::Arr(self.inner.clone(), self.ndim);
                    let selected = crate::functions::select_inner(&mask, v, current, self.inner.dtype())?;
                    ops::assign(&mut self.inner, selected)
                }
                Advanced::Take(idx, _) => {
                    let idx = Bound::new(key.py(), PyArr::new(Inner::U32(idx), 1))?;
                    self.put(idx.as_any(), value)
                }
            };
        }
        let basic = parse_basic(key, self)?;
        let value = value_of(value, self)?;
        if basic.steps.is_empty() {
            let v = match value {
                Value::Arr(i) => {
                    if i.len() == self.inner.len() && i.shape() != self.inner.shape() {
                        i.reshape(self.inner.shape())
                    } else {
                        i
                    }
                }
                Value::Num(n) => convert::full(&self.inner.context(), self.inner.shape(), self.inner.dtype(), n)?,
            };
            return ops::assign(&mut self.inner, v);
        }
        write_steps(&mut self.inner, &basic.steps, value)
    }

    /// Overwrites the whole array in place with `value` (an array that broadcasts, or a scalar);
    /// every handle on the data sees it.
    fn assign(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let ellipsis = PyEllipsis::get(value.py()).to_owned().into_any();
        self.__setitem__(&ellipsis, value)
    }
}
