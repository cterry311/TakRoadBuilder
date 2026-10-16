// src/python.rs
use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyRuntimeError};
use crate::game::{Tak, Move, GameResult};

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

    fn unmake_move(&mut self) {
        self.inner.unmake_move();
    }

    fn legal_moves(&self) -> Vec<String> {
        let mut buf = Vec::new();
        self.inner.get_all_legal_moves(&mut buf);
        buf.iter().map(|m| m.to_ptn()).collect()
    }

    fn filtered_moves(&mut self, depth: u8, do_pruning: bool) -> (Vec<String>, i8) {
        let (moves, score) = self.inner.get_filtered_moves(depth, do_pruning);
        (moves.iter().map(|m| m.to_ptn()).collect(), score)
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

    fn clone(&self) -> PyTak {
        PyTak { inner: self.inner.clone() }
    }
}