use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, Gamma};

use std::collections::HashMap;

use crate::game::{Tak, Move, GameResult, transform_move};

const POLICY_LEN: usize = 251 * 36;
const INPUT_LEN: usize = 25 * 36;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Proven { Unknown, Win, Loss, Draw }

#[derive(Clone)]
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
    key: Option<(u64, usize)>, // (hash, symmetry) of the leaf, when the eval cache is on
}

/// One model evaluation, stored in the canonical orientation of the position
/// (the symmetry `Tak::to_hash` picked), so any symmetric copy can reuse it.
struct CacheEntry {
    priors: Vec<(u16, f32)>, // (canonical move raw, prior) for every legal move, sorted by move
    value: f32,              // p_win - p_loss for the side to move
    last_used: u32,          // `EvalCache::age` when inserted or last hit
}

/// Model evaluations keyed by `Tak::to_hash`, so a position reached again by another
/// move order (or as a mirror / rotation) skips the model. Entries not used for more
/// than `max_age` rebases are dropped.
struct EvalCache {
    map: HashMap<u64, CacheEntry>,
    max_age: u32,
    age: u32,                // number of rebases so far
    hits: u64,
    misses: u64,
}

impl EvalCache {
    fn new(max_age: u32) -> EvalCache {
        EvalCache { map: HashMap::new(), max_age, age: 0, hits: 0, misses: 0 }
    }

    /// Priors in the order of `legal`, plus the value, if this position was evaluated before.
    fn lookup(&mut self, hash: u64, sym: usize, legal: &[Move]) -> Option<(Vec<f32>, f32)> {
        let age = self.age;
        let found = self.map.get_mut(&hash).and_then(|e| {
            if e.priors.len() != legal.len() { return None; }   // hash collision
            let mut priors = Vec::with_capacity(legal.len());
            for &mv in legal {
                let canon = transform_move(mv, sym).raw();
                let k = e.priors.binary_search_by_key(&canon, |&(m, _)| m).ok()?;
                priors.push(e.priors[k].1);
            }
            e.last_used = age;
            Some((priors, e.value))
        });
        if found.is_some() { self.hits += 1 } else { self.misses += 1 }
        found
    }

    fn insert(&mut self, hash: u64, sym: usize, legal: &[Move], priors: &[f32], value: f32) {
        let mut canon: Vec<(u16, f32)> = legal.iter().zip(priors)
            .map(|(&mv, &p)| (transform_move(mv, sym).raw(), p))
            .collect();
        canon.sort_unstable_by_key(|&(m, _)| m);
        self.map.insert(hash, CacheEntry { priors: canon, value, last_used: self.age });
    }

    fn on_rebase(&mut self) {
        self.age += 1;
        let (age, max_age) = (self.age, self.max_age);
        self.map.retain(|_, e| age - e.last_used <= max_age);
    }
}

pub struct Tree {
    nodes: Vec<Node>,    // arena, index 0 is the root
    root: Tak,           // the game being searched; mutated and restored during descent
    urgency: f32,        // FPU offset added to the parent's q for unvisited children;
    // positive values makes more likely to explore unexplored moves (bredth first), negative to prefer exploring nodes that are already explored (depth first). 0 will add no additional incentive
    c_puct: f32,         // a coefficent > 0 that controls exploration vs exploration, a larger coefficent favors branches that look very favorable but are likely already explored, lower value favors exploring boadly, including branches that may look less favorable
    pending: Vec<Pending>,// the set of pending leaves, waiting for input from the network to set the leaves value and policy
    cache: Option<EvalCache>, // None = eval cache disabled
}

impl Tree {
    pub fn new(root: Tak, urgency: f32, c_puct: f32) -> Tree {
        let root_node = Node {
            mv: Move::from_raw(0), prior: 1.0, n: 0, w: 0.0,
            proven: Proven::Unknown, pending: false, first_child: 0, n_children: 0,
        };
        Tree { nodes: vec![root_node], root, urgency, c_puct, pending: Vec::new(), cache: None }
    }

    /// Turn on the eval cache. Entries unused for more than `max_age` rebases are dropped.
    pub fn with_eval_cache(mut self, max_age: u32) -> Tree {
        self.cache = Some(EvalCache::new(max_age));
        self
    }

    /// (hits, misses, entries), all 0 when the cache is off.
    pub fn cache_stats(&self) -> (u64, u64, usize) {
        self.cache.as_ref().map_or((0, 0, 0), |c| (c.hits, c.misses, c.map.len()))
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

    /// Priors for `legal` from the full softmaxed policy row (POLICY_LEN), renormalized over legal moves.
    fn legal_priors(legal: &[Move], policy: &[f32]) -> Vec<f32> {
        let sum: f32 = legal.iter().map(|m| policy[m.policy_index()]).sum();
        legal.iter().map(|m| policy[m.policy_index()] / sum.max(1e-9)).collect()
    }

    /// Create all children for `idx`, `priors[i]` belongs to `legal[i]`.
    fn expand(&mut self, idx: usize, legal: &[Move], priors: &[f32]) {
        let first = self.nodes.len();
        for (&mv, &prior) in legal.iter().zip(priors) {
            self.nodes.push(Node {
                mv,
                prior,
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
                        let mut legal = Vec::new();
                        self.root.get_all_legal_moves(&mut legal);
                        let key = self.cache.as_ref().map(|_| self.root.to_hash());

                        // seen this position before: expand and back up right away, no model call
                        let hit = match (self.cache.as_mut(), key) {
                            (Some(cache), Some((hash, sym))) => cache.lookup(hash, sym, &legal),
                            _ => None,
                        };
                        if let Some((priors, value)) = hit {
                            self.expand(node, &legal, &priors);
                            self.backup(&path, value);
                        } else {
                            self.nodes[node].pending = true;
                            self.apply_virtual_loss(&path);
                            let row = self.pending.len();
                            self.root.encode_into(&mut buf[row * INPUT_LEN..(row + 1) * INPUT_LEN]);
                            let moves = path.len() - 1;
                            self.pending.push(Pending { path, legal, key });
                            for _ in 0..moves { self.root.unmake_move(); }
                            continue;
                        }
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
            let priors = Self::legal_priors(&p.legal, row);
            if let (Some(cache), Some((hash, sym))) = (self.cache.as_mut(), p.key) {
                cache.insert(hash, sym, &p.legal, &priors, values[i]);
            }
            self.expand(leaf, &p.legal, &priors);
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

    /// this method changes the root of the tree to the node assosiated with hte provided move from the root.
    /// Copies only that child's subtree into a new arena, renumbering `first_child` as it goes.
    /// Fails if a batch is pending (between collect and submit) or `mv` is not a child of the root.
    pub fn rebase_tree(&mut self, mv: Move) -> Result<(), String> {
        if !self.pending.is_empty() {
            return Err("rebase_tree called between collect and submit".to_string());
        }
        let root = &self.nodes[0];
        let old_idx = (root.first_child..root.first_child + root.n_children)
            .find(|&i| self.nodes[i].mv.raw() == mv.raw())
            .ok_or_else(|| format!("move {} is not a child of the root (root not expanded, or illegal move)", mv.to_ptn()))?;

        // breadth first copy: children are stored contiguously, so each node's children
        // land contiguously in the new arena and first_child can be rewritten as they are pushed
        let mut new_nodes = Vec::with_capacity(self.nodes.len());
        new_nodes.push(self.nodes[old_idx].clone());
        let mut head = 0;
        while head < new_nodes.len() {
            let (first, count) = (new_nodes[head].first_child, new_nodes[head].n_children);
            if count > 0 {
                new_nodes[head].first_child = new_nodes.len();
                for i in first..first + count {
                    new_nodes.push(self.nodes[i].clone());
                }
            }
            head += 1;
        }
        self.nodes = new_nodes;
        self.root.make_move(mv);
        if let Some(cache) = self.cache.as_mut() {
            cache.on_rebase();
        }
        Ok(())
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
#[cfg(test)]
mod tests {
    use super::*;

    /// Grow a tree with a uniform policy and value 0.
    fn grown_tree(rounds: usize) -> Tree {
        let mut tree = Tree::new(Tak::new(), -0.2, 1.5);
        let mut buf = vec![0.0f32; 64 * INPUT_LEN];
        for _ in 0..rounds {
            let n = tree.collect(&mut buf, 64);
            tree.submit(&vec![1.0; n * POLICY_LEN], &vec![0.0; n]);
        }
        tree
    }

    /// Same moves, visits and values in the same order, all the way down.
    fn same_subtree(a: &Tree, ia: usize, b: &Tree, ib: usize) {
        let (na, nb) = (&a.nodes[ia], &b.nodes[ib]);
        assert_eq!(na.mv.raw(), nb.mv.raw());
        assert_eq!((na.n, na.w, na.n_children), (nb.n, nb.w, nb.n_children));
        for k in 0..na.n_children {
            same_subtree(a, na.first_child + k, b, nb.first_child + k);
        }
    }

    #[test]
    fn rebase_keeps_exactly_the_subtree() {
        let old = grown_tree(40);
        let mut tree = grown_tree(40);
        let root = &tree.nodes[0];
        // the most visited child, so it has a real subtree
        let child = (root.first_child..root.first_child + root.n_children)
            .max_by_key(|&i| tree.nodes[i].n).unwrap();
        let mv = tree.nodes[child].mv;
        assert!(tree.nodes[child].n_children > 0);

        tree.rebase_tree(mv).unwrap();
        same_subtree(&old, child, &tree, 0);

        let mut expected = old.root.clone();
        expected.make_move(mv);
        assert_eq!(tree.root.to_tps(), expected.to_tps());

        // only the subtree is kept: count it in the old arena
        fn count(t: &Tree, i: usize) -> usize {
            let n = &t.nodes[i];
            1 + (n.first_child..n.first_child + n.n_children).map(|c| count(t, c)).sum::<usize>()
        }
        assert_eq!(tree.nodes.len(), count(&old, child));

        // the rebased tree keeps searching normally
        let mut buf = vec![0.0f32; 64 * INPUT_LEN];
        let before = tree.sims();
        let n = tree.collect(&mut buf, 64);
        tree.submit(&vec![1.0; n * POLICY_LEN], &vec![0.0; n]);
        assert!(tree.sims() > before);
    }

    #[test]
    fn rebase_rejects_bad_calls() {
        // root not expanded yet
        let mut fresh = Tree::new(Tak::new(), -0.2, 1.5);
        assert!(fresh.rebase_tree(Move::from_ptn("a1")).is_err());

        // between collect and submit
        let mut tree = grown_tree(5);
        let mut buf = vec![0.0f32; 64 * INPUT_LEN];
        tree.collect(&mut buf, 64);
        let mv = tree.nodes[tree.nodes[0].first_child].mv;
        assert!(tree.rebase_tree(mv).is_err());
    }

    /// Uneven policy, so priors differ per move and mix-ups show up.
    fn fake_policy(n: usize) -> Vec<f32> {
        (0..n * POLICY_LEN).map(|i| ((i * 7919) % 101 + 1) as f32).collect()
    }

    #[test]
    fn cache_maps_priors_through_symmetry() {
        let t = Tak::from_tps("x,2,x4/x,212S,2,1,x2/2,21C,21,1,1,1/2,x,2,x,1,1/2,1,22,1212C,121S,2/212S,x,1,1,x,1 1 30").unwrap();
        let mut legal = Vec::new();
        t.get_all_legal_moves(&mut legal);
        let priors = Tree::legal_priors(&legal, &fake_policy(1));
        let (hash, sym) = t.to_hash();
        let mut cache = EvalCache::new(2);
        cache.insert(hash, sym, &legal, &priors, 0.25);

        for s in 0..8 {
            // the same position turned by symmetry s
            let mut u = t.clone();
            for y in 0..6 {
                for x in 0..6 {
                    let sq = y * 8 + x;
                    u.stacks[crate::game::transform_sq(sq, s)] = t.stacks[sq];
                }
            }
            for y in 0..6 { for x in 0..6 { u.refresh_square(y * 8 + x); } }
            let (u_hash, u_sym) = u.to_hash();
            assert_eq!(u_hash, hash);
            let mut u_legal = Vec::new();
            u.get_all_legal_moves(&mut u_legal);

            let (got, value) = cache.lookup(u_hash, u_sym, &u_legal).expect("hit");
            assert_eq!(value, 0.25);
            // each move gets the prior of the move it corresponds to in the original
            for (k, &mv) in legal.iter().enumerate() {
                let i = u_legal.iter().position(|&m| m == crate::game::transform_move(mv, s)).unwrap();
                assert_eq!(got[i], priors[k], "symmetry {s}, move {}", mv.to_ptn());
            }
        }
        assert_eq!(cache.hits, 8);
    }

    #[test]
    fn cache_hits_and_eviction() {
        let mut tree = Tree::new(Tak::new(), -0.2, 1.5).with_eval_cache(1);
        let mut buf = vec![0.0f32; 64 * INPUT_LEN];
        for _ in 0..40 {
            let n = tree.collect(&mut buf, 64);
            tree.submit(&fake_policy(n), &vec![0.1; n]);
        }
        let (hits, misses, entries) = tree.cache_stats();
        assert!(hits > 0, "opening placements transpose, expected some hits");
        // every miss went to the model and was inserted; transpositions inside one batch share an entry
        assert!(entries > 0 && entries as u64 <= misses);

        // every visit is accounted for: root visits = model evals + cache hits (no terminals this early)
        assert_eq!(tree.sims() as u64, misses + hits);

        // entries unused for more than max_age (1) rebases are dropped
        let mv = tree.best_move();
        tree.rebase_tree(mv).unwrap();
        assert_eq!(tree.cache_stats().2, entries);       // age 1: all kept
        let n = tree.collect(&mut buf, 64);
        tree.submit(&fake_policy(n), &vec![0.1; n]);
        // age 2: only entries inserted or hit after the first rebase survive
        let recent = tree.cache.as_ref().unwrap().map.values().filter(|e| e.last_used >= 1).count();
        assert!(recent < tree.cache_stats().2);
        let mv = tree.best_move();
        tree.rebase_tree(mv).unwrap();
        assert_eq!(tree.cache_stats().2, recent);
    }

    #[test]
    fn cache_off_by_default() {
        let mut tree = grown_tree(5);
        let mv = tree.best_move();
        tree.rebase_tree(mv).unwrap();
        assert_eq!(tree.cache_stats(), (0, 0, 0));
    }
}
