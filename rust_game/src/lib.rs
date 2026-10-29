pub mod game;
mod python;
mod tree;

use pyo3::prelude::*;

#[pymodule]
fn tak_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<python::PyTak>()?;
    m.add_function(wrap_pyfunction!(python::encode_batch, m)?)?;
    m.add_function(wrap_pyfunction!(python::encode_training_batch, m)?)?;
    m.add_function(wrap_pyfunction!(python::from_tps_bulk, m)?)?;
    m.add_function(wrap_pyfunction!(python::raw_to_index, m)?)?;
    m.add_function(wrap_pyfunction!(python::bulk_raw_to_index, m)?)?;
    m.add_function(wrap_pyfunction!(python::index_to_raw, m)?)?;
    m.add_function(wrap_pyfunction!(python::index_to_ptn, m)?)?;
    m.add_class::<python::Search>()?;
    Ok(())
}