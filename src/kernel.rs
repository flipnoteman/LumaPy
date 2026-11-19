//! Custom WGSL kernels: `lumapy.Kernel`, `lumapy.view`, `lumapy.uniform`.

use luma::{BindingKind, KernelArg};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::array::PyArr;
use crate::context::PyContext;
use crate::convert::ctx_or_default;
use crate::dtype::{DType, Elem, PyNum};
use crate::each;
use crate::error::IntoPy;
use crate::inner::Inner;

/// A compiled custom compute kernel.
///
/// Luma binds `luma: LumaGrid` (`size: vec3<u32>`, the problem size) at `@group(0)
/// @binding(0)` and provides `luma_idx2(i, j)` / `luma_idx3(i, j, k)`. The kernel's own
/// resources are in `@group(1)`, numbered from 0 in the order of the arguments to `launch`.
/// Arrays bound `read_write` are written in place. Launches join the context's batch.
#[pyclass(name = "Kernel", module = "lumapy", frozen)]
pub struct PyKernel {
    inner: luma::Kernel,
}

/// A strided read view of an array passed to a kernel without a copy (binds two slots: the
/// buffer and a `LumaView` uniform). Made by `lumapy.view(array)`.
#[pyclass(name = "KernelView", module = "lumapy", frozen)]
pub struct PyView {
    inner: Inner,
}

/// A value passed to a kernel as a uniform: `Uniform(value, dtype=None)` is a `float32`
/// (`f32`), `uint32` (`u32`) or `int32` (`i32`) uniform. A bare Python float passed to
/// `launch` is an `f32`; a bare int a `u32` (or `i32` if negative).
#[pyclass(name = "Uniform", module = "lumapy", frozen)]
#[derive(Clone)]
pub struct PyUniform {
    value: UValue,
}

#[derive(Clone, Copy, Debug)]
enum UValue {
    F32(f32),
    U32(u32),
    I32(i32),
}

/// Passes `array` to a kernel as a strided read view (`luma_view_idx*` in WGSL).
#[pyfunction]
pub fn view(array: PyRef<'_, PyArr>) -> PyView {
    PyView { inner: array.inner.clone() }
}

#[pymethods]
impl PyUniform {
    #[new]
    #[pyo3(signature = (value, dtype = None))]
    fn new(value: &Bound<'_, PyAny>, dtype: Option<&Bound<'_, PyAny>>) -> PyResult<PyUniform> {
        uniform(value, dtype)
    }

    fn __repr__(&self) -> String {
        format!("lumapy.Uniform({:?})", self.value)
    }
}

fn uniform(value: &Bound<'_, PyAny>, dtype: Option<&Bound<'_, PyAny>>) -> PyResult<PyUniform> {
    let n = PyNum::extract(value).ok_or_else(|| PyTypeError::new_err("uniform takes a number"))?;
    let dtype = match dtype {
        Some(d) => DType::from_py(d)?,
        None => match n {
            PyNum::Float(_) => DType::F32,
            PyNum::Int(i) if i < 0 => DType::I32,
            _ => DType::U32,
        },
    };
    let value = match dtype {
        DType::F32 => UValue::F32(f32::from_num(n)?),
        DType::U32 => UValue::U32(u32::from_num(n)?),
        DType::I32 => UValue::I32(i32::from_num(n)?),
        other => {
            return Err(PyTypeError::new_err(format!("a uniform is float32, uint32 or int32, not {}", other.name())));
        }
    };
    Ok(PyUniform { value })
}

enum Holder {
    Array(Inner, bool),
    View(Inner),
    Uniform(UValue),
}

fn arg(holder: &mut Holder) -> KernelArg<'_> {
    match holder {
        Holder::Array(inner, true) => each!(inner, a => KernelArg::from(a)),
        Holder::Array(inner, false) => each!(inner, a => KernelArg::from(&*a)),
        Holder::View(inner) => each!(inner, a => luma::view(&*a)),
        Holder::Uniform(UValue::F32(v)) => luma::uniform(v),
        Holder::Uniform(UValue::U32(v)) => luma::uniform(v),
        Holder::Uniform(UValue::I32(v)) => luma::uniform(v),
    }
}

impl PyKernel {
    fn holders(&self, args: &Bound<'_, PyTuple>) -> PyResult<Vec<Holder>> {
        let bindings = self.inner.bindings();
        let mut slot = 0usize;
        let mut out = Vec::with_capacity(args.len());
        for item in args.iter() {
            if let Ok(a) = item.cast::<PyArr>() {
                let writable = bindings.get(slot).is_some_and(|b| b.kind == BindingKind::StorageReadWrite);
                // A clone shares the buffer, so writes through it reach the caller's array.
                out.push(Holder::Array(a.try_borrow()?.inner.clone(), writable));
                slot += 1;
            } else if let Ok(v) = item.cast::<PyView>() {
                out.push(Holder::View(v.get().inner.clone()));
                slot += 2;
            } else if let Ok(u) = item.cast::<PyUniform>() {
                out.push(Holder::Uniform(u.get().value));
                slot += 1;
            } else if PyNum::extract(&item).is_some() {
                out.push(Holder::Uniform(uniform(&item, None)?.value));
                slot += 1;
            } else {
                return Err(PyTypeError::new_err(format!(
                    "kernel arguments are lumapy arrays, lumapy.view(..), lumapy.uniform(..) or numbers, not {}",
                    item.get_type().name()?
                )));
            }
        }
        Ok(out)
    }
}

fn size3(obj: &Bound<'_, PyAny>) -> PyResult<Vec<u32>> {
    let v: Vec<u32> = match obj.extract::<u32>() {
        Ok(n) => vec![n],
        Err(_) => obj.extract()?,
    };
    if v.is_empty() || v.len() > 3 {
        return Err(PyValueError::new_err("a size has one to three extents"));
    }
    Ok(v)
}

#[pymethods]
impl PyKernel {
    /// Compiles `source` (WGSL) with entry point `entry`. `workgroup` overrides the shader's
    /// `@workgroup_size`; `label` names it in profiles. Building the same source twice reuses
    /// the pipeline.
    #[new]
    #[pyo3(signature = (source, entry = "main", workgroup = None, label = None, ctx = None))]
    fn new(
        py: Python<'_>,
        source: &str,
        entry: &str,
        workgroup: Option<[u32; 3]>,
        label: Option<&str>,
        ctx: Option<PyRef<'_, PyContext>>,
    ) -> PyResult<PyKernel> {
        let ctx = ctx_or_default(ctx.as_deref().map(|c| &c.inner))?;
        let inner = py
            .detach(|| {
                let mut b = ctx.kernel(source).entry(entry);
                if let Some(w) = workgroup {
                    b = b.workgroup(w);
                }
                if let Some(l) = label {
                    b = b.label(l);
                }
                b.build()
            })
            .py()?;
        Ok(PyKernel { inner })
    }

    /// The kernel's `@group(1)` bindings in order, as dicts with `name`, `kind` (`"read"`,
    /// `"read_write"`, `"uniform"`), `element_stride` and `min_size`.
    #[getter]
    fn bindings<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        self.inner
            .bindings()
            .iter()
            .map(|b| {
                let d = PyDict::new(py);
                d.set_item("name", &b.name)?;
                d.set_item(
                    "kind",
                    match b.kind {
                        BindingKind::StorageRead => "read",
                        BindingKind::StorageReadWrite => "read_write",
                        BindingKind::Uniform => "uniform",
                    },
                )?;
                d.set_item("element_stride", b.element_stride)?;
                d.set_item("min_size", b.min_size)?;
                Ok(d)
            })
            .collect()
    }

    /// The workgroup size.
    #[getter]
    fn workgroup(&self) -> [u32; 3] {
        self.inner.workgroup()
    }

    /// Records a launch with at least `size` invocations (an int or up to three extents;
    /// guard against `global_invocation_id >= luma.size` in WGSL).
    #[pyo3(signature = (size, *args))]
    fn launch(&self, size: &Bound<'_, PyAny>, args: &Bound<'_, PyTuple>) -> PyResult<()> {
        let size = size3(size)?;
        let mut holders = self.holders(args)?;
        let kargs: Vec<KernelArg<'_>> = holders.iter_mut().map(arg).collect();
        match *size.as_slice() {
            [x] => self.inner.launch([x], kargs),
            [x, y] => self.inner.launch([x, y], kargs),
            [x, y, z] => self.inner.launch([x, y, z], kargs),
            _ => unreachable!(),
        }
        .py()
    }

    /// Records a launch of exactly `workgroups` workgroups (three extents).
    #[pyo3(signature = (workgroups, *args))]
    fn launch_raw(&self, workgroups: [u32; 3], args: &Bound<'_, PyTuple>) -> PyResult<()> {
        let mut holders = self.holders(args)?;
        let kargs: Vec<KernelArg<'_>> = holders.iter_mut().map(arg).collect();
        self.inner.launch_raw(workgroups, kargs).py()
    }
}
