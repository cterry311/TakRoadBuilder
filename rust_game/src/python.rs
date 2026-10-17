// src/python.rs
use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyRuntimeError};
use crate::game::{Tak, Move, GameResult};

use numpy::{PyReadwriteArrayDyn, PyArray1, PyArray3, PyArrayMethods};

use crate::game::TENSOR_LEN;

#[pyclass(name = "Tak")]
pub struct PyTak {
    inner: Tak,
}

#[pymethods]
impl PyTak {
    #[new]
    fn new() -> Self {
        PyTak { inner: Tak::new() }
    }

    #[staticmethod]
    fn from_tps(tps: &str) -> PyResult<PyTak> {
        Tak::from_tps(tps)
            .map(|inner| PyTak { inner })
            .map_err(PyValueError::new_err)
    }

    fn to_tps(&self) -> String {
        self.inner.to_tps()
    }

    fn make_move(&mut self, ptn: &str) -> PyResult<()> {
        if self.inner.result != GameResult::Ongoing {
            return Err(PyRuntimeError::new_err("game is over"));
        }
        let mv = Move::from_ptn(ptn);          // ideally returns Result; see below
        self.inner.make_move(mv);
        Ok(())
    }

    fn make_move_raw(&mut self, mv: u16) -> PyResult<()> {
        if self.inner.result != GameResult::Ongoing {
            return Err(PyRuntimeError::new_err("game is over"));
        }
        let mv = Move::from_raw(mv);
        self.inner.make_move(mv);
        Ok(())
    }

    fn unmake_move(&mut self) {
        self.inner.unmake_move();
    }

    fn legal_moves(&self) -> Vec<String> {
        let mut buf = Vec::new();
        self.inner.get_all_legal_moves(&mut buf);
        buf.iter().map(|m| m.to_ptn()).collect()
    }
    
    fn legal_moves_raw(&self) -> Vec<u16> {
        let mut buf = Vec::new();
        self.inner.get_all_legal_moves(&mut buf);
        buf.iter().map(|m| m.raw()).collect()
    }

    /// gets filtered moves
    fn filtered_moves(&mut self, depth: u8, do_pruning: bool) -> (Vec<String>, i8) {
        let (moves, score) = self.inner.get_filtered_moves(depth, do_pruning);
        (moves.iter().map(|m| m.to_ptn()).collect(), score)
    }
    
    fn filtered_moves_raw(&mut self, depth: u8, do_pruning: bool) -> (Vec<u16>, i8) {
        let (moves, score) = self.inner.get_filtered_moves(depth, do_pruning);
        (moves.iter().map(|m| m.raw()).collect(), score)
    }

    /// 0 ongoing, 1 white win, 2 black win, 3 draw
    fn result(&self) -> u8 {
        match self.inner.result {
            GameResult::Ongoing => 0,
            GameResult::WhiteWin => 1,
            GameResult::BlackWin => 2,
            GameResult::Draw => 3,
        }
    }

    /// Fill `out` (float32, C-contiguous, 25*6*6 elements) with the encoded position.
    fn encode_into(&self, mut out: PyReadwriteArrayDyn<f32>) -> PyResult<()> {
        let slice = out
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("array must be C-contiguous"))?;
        if slice.len() != TENSOR_LEN {
            return Err(PyValueError::new_err(format!(
                "expected {TENSOR_LEN} elements, got {}", slice.len())));
        }
        self.inner.encode_into(slice);
        Ok(())
    }

    fn to_tensor<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<f32>>> {
        let t = self.inner.to_tensor();                  // [f32; 900]
        let flat = PyArray1::from_slice(py, &t);   // copies into a numpy array
        flat.reshape([25, 6, 6])
    }

    fn clone(&self) -> PyTak {
        PyTak { inner: self.inner.clone() }
    }
    
    #[staticmethod]
    fn ptn_to_raw(ptn: &str) -> PyResult<u16> {
        let raw = Move::from_ptn(ptn).raw();
        Ok(raw)
    }
    
    #[staticmethod]
    fn raw_to_ptn(raw: u16) -> PyResult<String> {
        let ptn = Move::from_raw(raw).to_ptn();
        Ok(ptn)
    }
}