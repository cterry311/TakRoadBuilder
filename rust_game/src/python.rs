// src/python.rs
use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyRuntimeError};
use crate::game::{Tak, Move, GameResult};
use crate::tree::{Tree};

use numpy::{PyReadwriteArrayDyn, PyArray1, PyArray3, PyArrayMethods, PyReadwriteArray1, PyReadonlyArray1};

use crate::game::TENSOR_LEN;

#[pyclass(name = "Tak")]
pub struct PyTak {
    inner: Tak,
}

#[pymethods]
impl PyTak {
    #[new]
    #[pyo3(signature = (komi = 0))]
    fn new(komi: u8) -> Self {
        PyTak { inner: Tak::new_with_komi(komi) }
    }

    #[staticmethod]
    #[pyo3(signature = (tps, komi = 0))]
    fn from_tps(tps: &str, komi: u8) -> PyResult<PyTak> {
        Tak::from_tps_with_komi(tps, komi)
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

    /// Returns a new uint8 array of shape (251, 6, 6); 1 marks a legal move.
    fn get_legal_mask<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<u8>>> {
        let mut scratch: Vec<Move> = Vec::with_capacity(256);
        let mut buf = vec![0u8; POLICY_LEN];
        self.inner.legal_mask(&mut scratch, &mut buf);
        PyArray1::from_slice(py, &buf)      // from_slice on numpy 0.23+
            .reshape([251, 6, 6])
            .map_err(Into::into)
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

#[pyfunction]
#[pyo3(signature = (tps_strs, komi = 0))]
pub fn from_tps_bulk(tps_strs: Vec<String>, komi: u8) -> PyResult<Vec<PyTak>> {
    tps_strs
        .iter()
        .map(|t| {
            Tak::from_tps_with_komi(t, komi)
                .map(|inner| PyTak { inner })
                .map_err(PyValueError::new_err)
        })
        .collect()
}

const POLICY_LEN: usize = 251 * 6 * 6;
const INPUT_LEN: usize = 25 * 36;
#[pyfunction]
#[pyo3(signature = (tps_strs, tensor, mask, komi = 0))]
pub fn encode_training_batch(
    tps_strs: Vec<String>,
    mut tensor: PyReadwriteArrayDyn<f32>,
    mut mask: PyReadwriteArrayDyn<u8>,
    komi: u8,
) -> PyResult<()> {
    let t = tensor
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("tensor must be C-contiguous"))?;
    let m = mask
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("mask must be C-contiguous"))?;
    if t.len() != tps_strs.len() * TENSOR_LEN || m.len() != tps_strs.len() * POLICY_LEN {
        return Err(PyValueError::new_err("buffer sizes do not match batch"));
    }
    let mut scratch: Vec<Move> = Vec::with_capacity(256);
    for ((tps, tc), mc) in tps_strs
        .iter()
        .zip(t.chunks_exact_mut(TENSOR_LEN))
        .zip(m.chunks_exact_mut(POLICY_LEN))
    {
        let tak = Tak::from_tps_with_komi(tps, komi).map_err(PyValueError::new_err)?;
        tak.encode_into(tc);
        tak.legal_mask(&mut scratch, mc);
    }
    Ok(())
}

/// Encode many games into one (batch, 25, 6, 6) float32 array.
#[pyfunction]
pub fn encode_batch(
    games: Vec<PyRef<'_, PyTak>>,
    mut out: PyReadwriteArrayDyn<f32>,
) -> PyResult<()> {
    let slice = out
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("array must be C-contiguous"))?;
    if slice.len() != games.len() * TENSOR_LEN {
        return Err(PyValueError::new_err(format!(
            "buffer has {} elements, expected {}",
            slice.len(),
            games.len() * TENSOR_LEN
        )));
    }
    for (g, chunk) in games.iter().zip(slice.chunks_exact_mut(TENSOR_LEN)) {
        g.inner.encode_into(chunk);
    }
    Ok(())
}

// in python.rs
#[pyfunction]
pub fn raw_to_index(raw: u16) -> usize {
    Move::from_raw(raw).policy_index()
}

#[pyfunction]
pub fn bulk_raw_to_index(raws: Vec<u16>) -> Vec<usize> {
    raws.into_iter()
        .map(|r| Move::from_raw(r).policy_index())
        .collect()
}

// python.rs
#[pyfunction]
pub fn index_to_raw(index: usize) -> PyResult<u16> {
    if index >= POLICY_LEN {
        return Err(PyValueError::new_err(format!(
            "policy index {index} out of range (max {})", POLICY_LEN - 1
        )));
    }
    Ok(Move::from_policy_index(index).raw())
}

#[pyfunction]
pub fn index_to_ptn(index: usize) -> PyResult<String> {
    if index >= POLICY_LEN {
        return Err(PyValueError::new_err(format!(
            "policy index {index} out of range (max {})", POLICY_LEN - 1
        )));
    }
    Ok(Move::from_policy_index(index).to_ptn())
}







#[pyclass(unsendable)]            // unsendable avoids Send requirements; fine for single-threaded use
pub struct Search { tree: Tree }

#[pymethods]
impl Search {
    #[new]
    #[pyo3(signature = (tps, urgency = -0.2, c_puct = 1.5, komi = 0))]
    fn new(tps: &str, urgency: f32, c_puct: f32, komi: u8) -> PyResult<Self> {
        let game = Tak::from_tps_with_komi(tps, komi).map_err(PyValueError::new_err)?;
        Ok(Search { tree: Tree::new(game, urgency, c_puct) })
    }

    fn collect(&mut self, mut buf: PyReadwriteArray1<f32>, max_leaves: usize) -> PyResult<usize> {
        let slice = buf.as_slice_mut().map_err(|e| PyValueError::new_err(e.to_string()))?;
        if slice.len() < max_leaves * INPUT_LEN {
            return Err(PyValueError::new_err("buffer too small"));
        }
        Ok(self.tree.collect(slice, max_leaves))
    }

    fn submit(&mut self, policy: PyReadonlyArray1<f32>, values: PyReadonlyArray1<f32>) -> PyResult<()> {
        let p = policy.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
        let v = values.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
        let n = self.tree.pending_len();
        if p.len() != n * POLICY_LEN || v.len() != n {
            return Err(PyValueError::new_err("policy/values length does not match pending leaves"));
        }
        self.tree.submit(p, v);
        Ok(())
    }

    fn sims(&self) -> u32 { self.tree.sims() }
    fn is_over(&self) -> bool { self.tree.is_over() }
    fn best_move_raw(&self) -> u16 { self.tree.best_move().raw() }
    fn root_stats(&self) -> Vec<(u16, u32, f32)> { self.tree.root_stats() }

    fn root_q(&self) -> f32 { self.tree.root_q() }

    fn add_noise(&mut self, alpha: f32, eps: f32, seed: u64) -> PyResult<()> {
        self.tree.add_root_noise(alpha, eps, seed);
        Ok(())
    }

    fn rebase_tree(&mut self, mv: u16) -> PyResult<()> {
        self.tree.rebase_tree(Move::from_raw(mv)).map_err(PyValueError::new_err)
    }

    fn to_string(&self) -> String { self.tree.to_string() }
}

