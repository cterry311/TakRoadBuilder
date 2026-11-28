// src/python.rs
use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyRuntimeError};
use crate::game::{Tak, Move, GameResult, transform_move, inverse_sym, Color};
use crate::tree::{Tree, collect_many, submit_many};

use numpy::{PyReadwriteArrayDyn, PyArray1, PyArray3, PyArrayMethods, PyReadwriteArray1, PyReadonlyArray1};

use crate::game::TENSOR_LEN;

#[pyclass(name = "Tak")]
pub struct PyTak {
    inner: Tak,
}

#[pymethods]
impl PyTak {

    /// creation of the class
    /// defaults komi argument to 0
    #[new]
    #[pyo3(signature = (komi = 0))]
    fn new(komi: u8) -> Self {
        PyTak { inner: Tak::new_with_komi(komi) }
    }

    /// creation of a `PyTak` instance from a tps string
    /// default argument for `komi` is 0
    #[staticmethod]
    #[pyo3(signature = (tps, komi = 0))]
    fn from_tps(tps: &str, komi: u8) -> PyResult<PyTak> {
        Tak::from_tps_with_komi(tps, komi)
            .map(|inner| PyTak { inner })
            .map_err(PyValueError::new_err)
    }

    /// returns a tps string representing the current state
    fn to_tps(&self) -> String {
        self.inner.to_tps()
    }

    /// make a move transforming and advancing the current state
    /// accepts a ptn string for the move
    fn make_move(&mut self, ptn: &str) -> PyResult<()> {
        if self.inner.result != GameResult::Ongoing {
            return Err(PyRuntimeError::new_err("game is over"));
        }
        let mv = Move::from_ptn(ptn);          // ideally returns Result; see below
        self.inner.make_move(mv);
        Ok(())
    }

    /// make a move transforming and advancing the current state
    /// accepts a raw integer to encoding the move to make
    fn make_move_raw(&mut self, mv: u16) -> PyResult<()> {
        if self.inner.result != GameResult::Ongoing {
            return Err(PyRuntimeError::new_err("game is over"));
        }
        let mv = Move::from_raw(mv);
        self.inner.make_move(mv);
        Ok(())
    }

    /// undoes the last move made
    fn unmake_move(&mut self) {
        self.inner.unmake_move();
    }

    /// gets a vector of all legal moves
    fn legal_moves(&self) -> Vec<String> {
        let mut buf = Vec::new();
        self.inner.get_all_legal_moves(&mut buf);
        buf.iter().map(|m| m.to_ptn()).collect()
    }

    /// gets a vector of all legal moves as raw integers
    fn legal_moves_raw(&self) -> Vec<u16> {
        let mut buf = Vec::new();
        self.inner.get_all_legal_moves(&mut buf);
        buf.iter().map(|m| m.raw()).collect()
    }

    /// gets filtered moves, filtered for the best class provable within a given depth
    fn filtered_moves(&mut self, depth: u8, do_pruning: bool) -> (Vec<String>, i8) {
        let (moves, score) = self.inner.get_filtered_moves(depth, do_pruning);
        (moves.iter().map(|m| m.to_ptn()).collect(), score)
    }

    /// gets filtered moves, filtered for the best class provable within a given depth, as raw integers
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

    /// returns a numpy array encoding game state
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

    /// create a copy of a `PyTak` instance
    fn clone(&self) -> PyTak {
        PyTak { inner: self.inner.clone() }
    }

    /// convert a ptn string to a raw integer encoding
    #[staticmethod]
    fn ptn_to_raw(ptn: &str) -> PyResult<u16> {
        let raw = Move::from_ptn(ptn).raw();
        Ok(raw)
    }

    /// convert a raw integer encoding to a ptn string
    #[staticmethod]
    fn raw_to_ptn(raw: u16) -> PyResult<String> {
        let ptn = Move::from_raw(raw).to_ptn();
        Ok(ptn)
    }

    /// convert the game to a hash
    fn __hash__(&self) -> u64 {
        self.inner.to_hash().0
    }

    /// get the rotation/reflection of the hash
    fn hash_transform(&self) -> usize {
        self.inner.to_hash().1
    }

    /// get the side to move, either "white" or "black"
    fn side_to_move(&self) -> String {
        match self.inner.side_to_move() {
            Color::White => "white".to_string(),
            Color::Black => "black".to_string(),
        }
    }
}

/// bulk convert a vector of tps strings to a vector of `PyTak` instances
/// `komi` is 0 by default
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

/// Encode a batch of tps strings to a batch of input tensors as well as a mask of legal moves
/// `komi` is 0 by default.
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
/// convert a raw move integer, into it's index in a policy output tensor
#[pyfunction]
pub fn raw_to_index(raw: u16) -> usize {
    Move::from_raw(raw).policy_index()
}

/// encode a batch of raw move integers into a batch of policy output indexes
#[pyfunction]
pub fn bulk_raw_to_index(raws: Vec<u16>) -> Vec<usize> {
    raws.into_iter()
        .map(|r| Move::from_raw(r).policy_index())
        .collect()
}

// python.rs
/// convert a policy output index into a raw move integer
#[pyfunction]
pub fn index_to_raw(index: usize) -> PyResult<u16> {
    if index >= POLICY_LEN {
        return Err(PyValueError::new_err(format!(
            "policy index {index} out of range (max {})", POLICY_LEN - 1
        )));
    }
    Ok(Move::from_policy_index(index).raw())
}


/// convert a policy output index into a ptn string
#[pyfunction]
pub fn index_to_ptn(index: usize) -> PyResult<String> {
    if index >= POLICY_LEN {
        return Err(PyValueError::new_err(format!(
            "policy index {index} out of range (max {})", POLICY_LEN - 1
        )));
    }
    Ok(Move::from_policy_index(index).to_ptn())
}


/// transform a ptn move into a reflected and rotated move
/// `sym` encodes the transformation
#[pyfunction]
pub fn transform_ptn(ptn: &str, sym: usize) -> String {
    let mv = Move::from_ptn(ptn);
    let t_move = transform_move(mv, sym);
    t_move.to_ptn()
}

/// get the inverse for the provided transformation such that the application of both transformations leads to the original state
#[pyfunction]
pub fn inverse_transform(sym: usize) -> usize {
    inverse_sym(sym)
}


#[pyclass]                        // Send, so collect_batch / submit_batch can hand trees to other threads
pub struct Search { tree: Tree }

/// `Search.collect` for many searches at once, run in parallel on all cores with the GIL released.
/// Rows are packed in search order with no gaps. Returns the row count of each search.
#[pyfunction]
pub fn collect_batch(
    py: Python<'_>,
    mut searches: Vec<PyRefMut<'_, Search>>,
    mut buf: PyReadwriteArray1<f32>,
    max_leaves: usize,
) -> PyResult<Vec<usize>> {
    let slice = buf.as_slice_mut().map_err(|e| PyValueError::new_err(e.to_string()))?;
    if slice.len() < searches.len() * max_leaves * INPUT_LEN {
        return Err(PyValueError::new_err(format!(
            "buffer too small: need {} rows for {} searches * {} leaves",
            searches.len() * max_leaves, searches.len(), max_leaves)));
    }
    let mut trees: Vec<&mut Tree> = searches.iter_mut().map(|s| &mut s.tree).collect();
    Ok(py.detach(|| collect_many(&mut trees, slice, max_leaves)))
}

/// `Search.submit` for many searches at once, run in parallel with the GIL released.
/// `searches` must be the same list, in the same order, as the last `collect_batch`.
#[pyfunction]
pub fn submit_batch(
    py: Python<'_>,
    mut searches: Vec<PyRefMut<'_, Search>>,
    policy: PyReadonlyArray1<f32>,
    values: PyReadonlyArray1<f32>,
) -> PyResult<()> {
    let p = policy.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let v = values.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let n: usize = searches.iter().map(|s| s.tree.pending_len()).sum();
    if p.len() != n * POLICY_LEN || v.len() != n {
        return Err(PyValueError::new_err(format!(
            "expected {n} rows (policy {} floats, values {n}), got policy {} and values {}",
            n * POLICY_LEN, p.len(), v.len())));
    }
    let mut trees: Vec<&mut Tree> = searches.iter_mut().map(|s| &mut s.tree).collect();
    py.detach(|| submit_many(&mut trees, p, v));
    Ok(())
}

#[pymethods]
impl Search {

    /// create a new search tree
    #[new]
    #[pyo3(signature = (tps, urgency = -0.2, c_puct = 1.5, komi = 0, eval_cache = true, cache_max_age = 2, capacity = 50_000))]
    fn new(tps: &str, urgency: f32, c_puct: f32, komi: u8, eval_cache: bool, cache_max_age: u32, capacity: usize) -> PyResult<Self> {
        let game = Tak::from_tps_with_komi(tps, komi).map_err(PyValueError::new_err)?;
        let mut tree = Tree::new(game, urgency, c_puct, capacity);
        if eval_cache {
            tree = tree.with_eval_cache(cache_max_age);
        }
        Ok(Search { tree })
    }

    /// collect a set of leaves to be expanded, write their tensor embeddings to the buffer
    fn collect(&mut self, mut buf: PyReadwriteArray1<f32>, max_leaves: usize) -> PyResult<usize> {
        let slice = buf.as_slice_mut().map_err(|e| PyValueError::new_err(e.to_string()))?;
        if slice.len() < max_leaves * INPUT_LEN {
            return Err(PyValueError::new_err("buffer too small"));
        }
        Ok(self.tree.collect(slice, max_leaves))
    }

    /// submit a set of policy and value outputs to the tree, fill out the backed up nodes in the tree with the given inputs
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

    /// get the numbers of sims at the root
    fn sims(&self) -> u32 { self.tree.sims() }

    /// check if the game in the tree is over
    fn is_over(&self) -> bool { self.tree.is_over() }

    /// get the best move as a raw integer
    fn best_move_raw(&self) -> u16 { self.tree.best_move().raw() }

    /// get the stats at the root (raw move, visits, q value)
    fn root_stats(&self) -> Vec<(u16, u32, f32)> { self.tree.root_stats() }

    /// get the q value at the root
    fn root_q(&self) -> f32 { self.tree.root_q() }

    /// add noise to the root node policy
    /// # Arguments
    /// * `alpha` - how spiky the noise is, lower values is more spiky
    /// * `eps` - how much of the policy to replace with noise
    /// * `seed` - the seed for the pseudorandom number generator
    fn add_noise(&mut self, alpha: f32, eps: f32, seed: u64) -> PyResult<()> {
        self.tree.add_root_noise(alpha, eps, seed);
        Ok(())
    }

    /// rebase the tree to a new starting position
    /// # Arguments
    /// * `mv` - the raw integer encoding for the move, the move to take from the root to rebase the tree
    fn rebase_tree(&mut self, mv: u16) -> PyResult<()> {
        self.tree.rebase_tree(Move::from_raw(mv)).map_err(PyValueError::new_err)
    }

    /// get the current tree as a string
    fn to_string(&self) -> String { self.tree.to_string() }

    /// (hits, misses, entries) for the eval cache, all 0 when it is off.
    fn cache_stats(&self) -> (u64, u64, usize) { self.tree.cache_stats() }

    /// get the average depth of the tree
    fn avg_depth(&self) -> f32 {
        self.tree.avg_depth()
    }

    /// get the max depth of the tree
    fn max_depth(&self) -> u8 {
        self.tree.max_depth()
    }


    /// get the stats of depth at the root (raw_move, avg depth, max depth)
    fn depth_stats(&self) -> Vec<(u16, f32, u16)> {
        self.tree.root_depths()
    }

    /// get the number of nodes in the tree
    fn node_count(&self) -> usize { self.tree.node_count() }

}

