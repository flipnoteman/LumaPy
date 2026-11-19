//! Luma errors as Python exceptions.

use pyo3::create_exception;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

create_exception!(
    lumapy,
    LumaError,
    PyRuntimeError,
    "An error reported by Luma: a failed GPU operation, a shape mismatch found while recording, a device that could not be opened, ... Operators do not raise it themselves: a failed operation yields a poisoned array, and the error comes out where the host needs a result (`to_numpy`, `item`, `eval`, `check`)."
);

/// Converts a Luma error into a `LumaError` exception carrying the full message (operation,
/// source location and cause).
pub fn luma_err(error: luma::LumaError) -> PyErr {
    LumaError::new_err(error.to_string())
}

/// `Result<T, luma::LumaError>` into `PyResult<T>`.
pub trait IntoPy<T> {
    fn py(self) -> PyResult<T>;
}

impl<T> IntoPy<T> for luma::Result<T> {
    fn py(self) -> PyResult<T> {
        self.map_err(luma_err)
    }
}
