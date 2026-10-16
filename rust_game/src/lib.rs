pub mod game;
mod python;

use pyo3::prelude::*;

#[pymodule]
fn tak_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<python::PyTak>()?;
    Ok(())
}