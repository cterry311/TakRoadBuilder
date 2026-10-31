use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, Gamma};

use crate::game::{Tak, Move, GameResult};

const POLICY_LEN: usize = 251 * 36;
const INPUT_LEN: usize = 25 * 36;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Proven { Unknown, Win, Loss, Draw }

struct Node {
    mv: Move,            // move that leads INTO this node
    prior: f32,          // policy probability from the parent's expansion
    n: u32,              // visits (u32 so it can't overflow at 65k)
    w: f32,              // sum of ONE backed-up value per simulation through this node,
    // from the view of the player to move AT this node
    proven: Proven,      // exact result for the player to move at this node
    pending: bool,       // claimed by a leaf in the current batch
    first_child: usize,  // the index of the first child in the tree Nodes array
    n_children: usize,   // 0 = not expanded yet
}

impl Node {
    fn q(&self) -> f32 {
        if self.n == 0 { 0.0 } else { self.w / self.n as f32 }
    }

    /// get a json representation of the node
    pub fn to_string(&self) -> String {
        format!("{{ \"ptn\": \"{}\", \"prior\": {}, \"n\": {}, \"w\": {}, \"q\": {}, \"first_child\": {}, \"n_children\": {} }}", self.mv.to_ptn(), self.prior, self.n, self.w, self.q(), self.first_child, self.n_children)
    }
}

struct Pending {
    path: Vec<usize>,    // node indices, root .. leaf
    legal: Vec<Move>,    // legal moves at the leaf, captured while the game was there
}

pub struct Tree {
    nodes: Vec<Node>,    // arena, index 0 is the root
    root: Tak,           // the game being searched; mutated and restored during descent
    urgency: f32,        // FPU offset added to the parent's q for unvisited children;
    // positive values makes more likely to explore unexplored moves (bredth first), negative to prefer exploring nodes that are already explored (depth first). 0 will add no additional incentive
    c_puct: f32,         // a coefficent > 0 that controls exploration vs exploration, a larger coefficent favors branches that look very favorable but are likely already explored, lower value favors exploring boadly, including branches that may look less favorable
    pending: Vec<Pending>,// the set of pending leaves, waiting for input from the network to set the leaves value and policy
}

impl Tree {
    pub fn new(root: Tak, urgency: f32, c_puct: f32) -> Tree {
        let root_node = Node {
            mv: Move::from_raw(0), prior: 1.0, n: 0, w: 0.0,
            proven: Proven::Unknown, pending: false, first_child: 0, n_children: 0,
        };
        Tree { nodes: vec![root_node], root, urgency, c_puct, pending: Vec::new() }
    }

    /// Returns the index of the chosen child of `idx`. Only call on an expanded node.
    fn select_child(&self, idx: usize) -> usize {
        let node = &self.nodes[idx];
        // invariant: node.n = 1 + sum(child.n); max(1) so the first pick uses the priors
        let sqrt_total = (node.n.saturating_sub(1) as f32).max(1.0).sqrt();
        let fpu = node.q() + self.urgency;       // parent's own view: NOT negated

        let mut best = node.first_child;
        let mut best_score = f32::NEG_INFINITY;
        for i in node.first_child..node.first_child + node.n_children {
            let c = &self.nodes[i];
            let q = if c.n > 0 { -c.q() } else { fpu };   // child's view is the opponent's: negate
            let u = self.c_puct * c.prior * sqrt_total / (1.0 + c.n as f32);
            if q + u > best_score {
                best_score = q + u;
                best = i;
            }
        }
        best
    }

    /// Create all children for `idx`. `policy` is the full softmaxed policy row (POLICY_LEN).
    fn expand(&mut self, idx: usize, legal: &[Move], policy: &[f32]) {
        let sum: f32 = legal.iter().map(|m| policy[m.policy_index()]).sum();
        let first = self.nodes.len();
        for &mv in legal {
            self.nodes.push(Node {
                mv,
                prior: policy[mv.policy_index()] / sum.max(1e-9),  // renormalize over legal moves
                n: 0, w: 0.0,
                proven: Proven::Unknown, pending: false,
                first_child: 0, n_children: 0,
            });
        }
        self.nodes[idx].first_child = first;
        self.nodes[idx].n_children = legal.len();
    }

    /// The only place values flip sign.
    fn backup(&mut self, path: &[usize], mut value: f32) {
        for &i in path.iter().rev() {
            let node = &mut self.nodes[i];
            node.n += 1;
            node.w += value;
            value = -value;
        }
    }

    // +1 on every node, NO alternation: each node's own q looks good,
    // so its parent (which negates) sees the path as bad.
    fn apply_virtual_loss(&mut self, path: &[usize]) {
        for &i in path { self.nodes[i].n += 1; self.nodes[i].w += 1.0; }
    }
    fn remove_virtual_loss(&mut self, path: &[usize]) {
        for &i in path { self.nodes[i].n -= 1; self.nodes[i].w -= 1.0; }
    }

    /// Descend up to `max_leaves` times, writing one encoding per pending leaf into `buf`.
    /// Returns the number of rows written.
    pub fn collect(&mut self, buf: &mut [f32], max_leaves: usize) -> usize {
        self.pending.clear();
        let mut attempts = 0;
        while self.pending.len() < max_leaves && attempts < 2 * max_leaves {
            attempts += 1;
            let mut path = vec![0usize];
            let mut node = 0usize;

            // SELECT
            while self.nodes[node].n_children > 0 && self.nodes[node].proven == Proven::Unknown {
                node = self.select_child(node);
                self.root.make_move(self.nodes[node].mv);
                path.push(node);
            }

            // first landing on a finished game: record the exact result
            if self.nodes[node].proven == Proven::Unknown
                && self.root.result != GameResult::Ongoing
            {
                let v = terminal_value(&self.root);
                self.nodes[node].proven =
                    if v > 0.0 { Proven::Win } else if v < 0.0 { Proven::Loss } else { Proven::Draw };
            }

            match self.nodes[node].proven {
                Proven::Win  => self.backup(&path,  1.0),
                Proven::Loss => self.backup(&path, -1.0),
                Proven::Draw => self.backup(&path,  0.0),
                Proven::Unknown => {
                    if !self.nodes[node].pending {
                        self.nodes[node].pending = true;
                        self.apply_virtual_loss(&path);
                        let row = self.pending.len();
                        self.root.encode_into(&mut buf[row * INPUT_LEN..(row + 1) * INPUT_LEN]);
                        let mut legal = Vec::new();
                        self.root.get_all_legal_moves(&mut legal);
                        self.pending.push(Pending { path, legal });
                        // (path moved into pending; unmake count computed below from its length)
                        let moves = self.pending.last().unwrap().path.len() - 1;
                        for _ in 0..moves { self.root.unmake_move(); }
                        continue;
                    }
                    // else: duplicate of a leaf already in this batch, discard
                }
            }
            for _ in 0..path.len() - 1 { self.root.unmake_move(); }
        }
        self.pending.len()
    }

    /// `policy`: n * POLICY_LEN softmaxed probabilities. `values`: n values (p_win - p_loss),
    /// mover-relative. Row i corresponds to pending[i].
    pub fn submit(&mut self, policy: &[f32], values: &[f32]) {
        let pending = std::mem::take(&mut self.pending);
        for (i, p) in pending.into_iter().enumerate() {
            let row = &policy[i * POLICY_LEN..(i + 1) * POLICY_LEN];
            let leaf = *p.path.last().unwrap();
            self.expand(leaf, &p.legal, row);
            self.nodes[leaf].pending = false;
            self.remove_virtual_loss(&p.path);
            self.backup(&p.path, values[i]);
        }
    }

    pub fn best_move(&self) -> Move {
        let root = &self.nodes[0];
        let mut best = root.first_child;
        for i in root.first_child..root.first_child + root.n_children {
            if self.nodes[i].n > self.nodes[best].n { best = i; }
        }
        self.nodes[best].mv
    }

    pub fn pending_len(&self) -> usize { self.pending.len() }
    pub fn sims(&self) -> u32 { self.nodes[0].n }        // valid between submit and collect, when virtual loss is cleared
    pub fn is_over(&self) -> bool { self.root.result != GameResult::Ongoing }


    pub fn root_stats(&self) -> Vec<(u16, u32, f32)> {    // (raw move, visits, -q) for logging and later training targets
        let root = &self.nodes[0];
        (root.first_child..root.first_child + root.n_children)
            .map(|i| (self.nodes[i].mv.raw(), self.nodes[i].n, -self.nodes[i].q()))
            .collect()
    }

    pub fn root_q(&self) -> f32 {
        self.nodes[0].q()
    }

    /// get a json representation of the search tree
    pub fn to_string(&self) -> String {
        let child_strs: Vec<String> = self.nodes.iter().map(|m| m.to_string()).collect();
        format!("{{ \"tps\": \"{}\", \"nodes\": [{}] }}", self.root.to_tps(), child_strs.join(", "))
    }

    pub fn add_root_noise(&mut self, alpha: f32, eps: f32, seed: u64) {
        // 1. Find the root's child range. Copy the numbers out so no borrow is held.
        let (start, end) = {
            let root = &self.nodes[0];
            (root.first_child, root.first_child + root.n_children)
        };
        if end <= start { return; }             // root not expanded yet: nothing to do

        // 2. Draw one Gamma(alpha, 1) sample per child.
        let mut rng = StdRng::seed_from_u64(seed);
        let gamma = Gamma::new(alpha, 1.0).unwrap();
        let mut noise: Vec<f32> = (start..end).map(|_| gamma.sample(&mut rng)).collect();

        // 3. Normalize so the samples sum to 1. That is a Dirichlet(alpha, ..., alpha) draw.
        let sum: f32 = noise.iter().sum();
        if !(sum.is_finite() && sum > 0.0) { return; }   // tiny alpha can give all zeros: skip
        for x in noise.iter_mut() { *x /= sum; }

        // 4. Mix into the priors.
        for (k, i) in (start..end).enumerate() {
            let p = self.nodes[i].prior;
            self.nodes[i].prior = (1.0 - eps) * p + eps * noise[k];
        }
    }
}

/// Value from the view of the side to move at `g`.
fn terminal_value(g: &Tak) -> f32 {
    let white_to_move = g.ply & 1 == 0;
    match g.result {
        GameResult::WhiteWin => if white_to_move {  1.0 } else { -1.0 },
        GameResult::BlackWin => if white_to_move { -1.0 } else {  1.0 },
        _ => 0.0,
    }
}