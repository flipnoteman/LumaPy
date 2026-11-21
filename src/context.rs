//! `lumapy.Context` (a device and its batch of recorded work) and GPU profiles.

use luma::{BackendKind, Context, ContextOptions, DeviceKind, Profile, TuningProfile};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::error::IntoPy;

/// A device (a GPU, or the host CPU) with its own batch of recorded operations.
///
/// Arrays created without `ctx=` live on the default context (`lumapy.default_context()`),
/// opened on first use. Options left as `None` come from the `LUMA_*` environment variables.
///
/// * `device`: `"auto"` (a GPU, else the CPU backend), `"gpu"` or `"cpu"`.
/// * `backend`: `"auto"` (CUDA, then Metal, then wgpu, then the CPU), `"cuda"`, `"metal"`, `"wgpu"`
///   or `"cpu"`.
/// * `memory_budget`: bytes of array data kept on the device before spilling to host memory.
/// * `chunk_bytes`: the largest buffer an array is split into (a power of two).
/// * `calibrate`: measure the device's throughputs when it opens (cached per machine).
/// * `check_finite`: fail any operation whose result holds a NaN or Inf (debug mode, slow).
/// * `fast_math`: allow rewrites that change values (`x * 0`, reassociation, ...).
/// * `df_fma`: whether `df32` products take their error from `fma`.
#[pyclass(name = "Context", module = "lumapy", frozen)]
#[derive(Clone)]
pub struct PyContext {
    pub inner: Context,
}

fn device_kind(name: &str) -> PyResult<DeviceKind> {
    match name.to_ascii_lowercase().as_str() {
        "auto" => Ok(DeviceKind::Auto),
        "gpu" => Ok(DeviceKind::Gpu),
        "cpu" => Ok(DeviceKind::Cpu),
        other => Err(PyValueError::new_err(format!("device is \"auto\", \"gpu\" or \"cpu\", not {other:?}"))),
    }
}

fn backend_kind(name: &str) -> PyResult<BackendKind> {
    match name.to_ascii_lowercase().as_str() {
        "auto" => Ok(BackendKind::Auto),
        "cuda" => Ok(BackendKind::Cuda),
        "metal" => Ok(BackendKind::Metal),
        "wgpu" => Ok(BackendKind::Wgpu),
        "cpu" => Ok(BackendKind::Cpu),
        other => Err(PyValueError::new_err(format!(
            "backend is \"auto\", \"cuda\", \"metal\", \"wgpu\" or \"cpu\", not {other:?}"
        ))),
    }
}

#[pymethods]
impl PyContext {
    #[new]
    #[pyo3(signature = (device = None, *, backend = None, memory_budget = None, chunk_bytes = None, calibrate = None, check_finite = None, fast_math = None, df_fma = None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        device: Option<&str>,
        backend: Option<&str>,
        memory_budget: Option<u64>,
        chunk_bytes: Option<u64>,
        calibrate: Option<bool>,
        check_finite: Option<bool>,
        fast_math: Option<bool>,
        df_fma: Option<bool>,
    ) -> PyResult<PyContext> {
        let mut options = ContextOptions::from_env();
        if let Some(d) = device {
            options = options.device(device_kind(d)?);
        }
        if let Some(b) = backend {
            options = options.backend(backend_kind(b)?);
        }
        if let Some(b) = memory_budget {
            options = options.memory_budget(b);
        }
        if let Some(b) = chunk_bytes {
            options = options.chunk_bytes(b);
        }
        if let Some(on) = calibrate {
            options = options.calibrate(on);
        }
        if let Some(on) = check_finite {
            options = options.check_finite(on);
        }
        if let Some(on) = fast_math {
            options = options.fast_math(on);
        }
        if let Some(on) = df_fma {
            options = options.df_fma(on);
        }
        let inner = py.detach(|| Context::with_options(options)).py()?;
        Ok(PyContext { inner })
    }

    /// The native CPU backend, whatever GPUs exist.
    #[staticmethod]
    fn cpu(py: Python<'_>) -> PyResult<PyContext> {
        Ok(PyContext { inner: py.detach(Context::cpu).py()? })
    }

    /// Whether this context runs on the CPU backend.
    #[getter]
    fn is_cpu(&self) -> bool {
        self.inner.is_cpu()
    }

    /// The backend this context runs on: `"cuda"`, `"metal"`, `"wgpu"` or `"cpu"`.
    #[getter]
    fn backend(&self) -> String {
        self.inner.backend().to_string()
    }

    /// The device's name (the GPU's name, or a description of the CPU).
    #[getter]
    fn device_name(&self) -> &str {
        self.inner.device_name()
    }

    /// Submits the recorded work and waits until the GPU has finished it.
    fn sync(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.sync()).py()
    }

    /// Hands the recorded work to the GPU without waiting for it.
    fn flush(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.flush()).py()
    }

    /// The kernel tuning profile in use: `"apple"`, `"nvidia"`, `"generic"` or `"cpu"`.
    #[getter]
    fn tuning_profile(&self) -> &'static str {
        match self.inner.tuning_profile() {
            TuningProfile::Apple => "apple",
            TuningProfile::Nvidia => "nvidia",
            TuningProfile::Generic => "generic",
            TuningProfile::Cpu => "cpu",
            _ => "unknown",
        }
    }

    /// The device features the context was opened with, by name (`"SHADER_F64"`, ...).
    #[getter]
    fn features(&self) -> Vec<String> {
        self.inner.features().iter_names().map(|(name, _)| name.to_owned()).collect()
    }

    /// Bytes of array data currently moved out to host memory because the device's memory
    /// budget is exhausted.
    #[getter]
    fn spilled_bytes(&self) -> u64 {
        self.inner.spilled_bytes()
    }

    /// Whether the adapter supports the timestamp queries `profile` needs.
    #[getter]
    fn profiling_supported(&self) -> bool {
        self.inner.profiling_supported()
    }

    /// Starts measuring the GPU time of every kernel launched from now on.
    fn enable_profiling(&self) -> PyResult<()> {
        self.inner.enable_profiling().py()
    }

    /// Stops measuring (waiting for the GPU) and returns the measurements.
    fn finish_profiling(&self, py: Python<'_>) -> PyResult<PyProfile> {
        Ok(PyProfile { inner: py.detach(|| self.inner.finish_profiling()).py()? })
    }

    /// Calls `fn()` with profiling on and returns `(result, Profile)`; waits for the GPU.
    fn profile(&self, py: Python<'_>, f: &Bound<'_, PyAny>) -> PyResult<(Py<PyAny>, PyProfile)> {
        let mut failure = None;
        let (result, profile) = self
            .inner
            .profile(|| match f.call0() {
                Ok(r) => Some(r.unbind()),
                Err(e) => {
                    failure = Some(e);
                    None
                }
            })
            .py()?;
        if let Some(e) = failure {
            return Err(e);
        }
        Ok((result.unwrap_or_else(|| py.None()), PyProfile { inner: profile }))
    }

    fn __repr__(&self) -> String {
        format!(
            "lumapy.Context(backend={:?}, device_name={:?}, tuning_profile={:?})",
            self.backend(),
            self.device_name(),
            self.tuning_profile()
        )
    }
}

/// The GPU time of one kernel, summed over its launches in a profile. Times are in seconds.
#[pyclass(name = "KernelTiming", module = "lumapy", frozen, get_all)]
#[derive(Clone)]
pub struct PyKernelTiming {
    /// The kernel's label (`Fused(Mul, Add, Sqrt) f32`, a custom kernel's label, ...).
    label: String,
    launches: u32,
    total: f64,
    min: f64,
    max: f64,
    mean: f64,
}

#[pymethods]
impl PyKernelTiming {
    fn __repr__(&self) -> String {
        format!(
            "KernelTiming(label={:?}, launches={}, total={:.3e}s, mean={:.3e}s)",
            self.label, self.launches, self.total, self.mean
        )
    }
}

/// GPU durations per kernel, measured with timestamp queries (see `Context.profile`).
#[pyclass(name = "Profile", module = "lumapy", frozen)]
pub struct PyProfile {
    pub inner: Profile,
}

#[pymethods]
impl PyProfile {
    /// One entry per kernel label, the one with the most total time first.
    #[getter]
    fn kernels(&self) -> Vec<PyKernelTiming> {
        self.inner
            .kernels
            .iter()
            .map(|k| PyKernelTiming {
                label: k.label.clone(),
                launches: k.launches,
                total: k.total.as_secs_f64(),
                min: k.min.as_secs_f64(),
                max: k.max.as_secs_f64(),
                mean: k.mean().as_secs_f64(),
            })
            .collect()
    }

    /// Where lazy expressions were cut into several kernels, as messages.
    #[getter]
    fn splits(&self) -> Vec<String> {
        self.inner.splits.iter().map(|s| s.to_string()).collect()
    }

    /// The entry of the kernel with this label, or `None`.
    fn get(&self, label: &str) -> Option<PyKernelTiming> {
        self.kernels().into_iter().find(|k| k.label == label)
    }

    /// Total GPU time of all measured launches, in seconds.
    #[getter]
    fn total(&self) -> f64 {
        self.inner.total().as_secs_f64()
    }

    /// Number of measured launches.
    #[getter]
    fn launches(&self) -> u32 {
        self.inner.launches()
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        format!(
            "lumapy.Profile(kernels={}, launches={}, total={:.3e}s)",
            self.inner.kernels.len(),
            self.launches(),
            self.total()
        )
    }
}
