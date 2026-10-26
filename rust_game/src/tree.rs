use crate::game::{Tak, Move, GameResult};



#[derive(Clone, Copy, PartialEq, Eq)]
enum Proven { Unknown, Win, Loss, Draw }

struct Node {
    mv: Move,            // the move that leads INTO this node (your Move(u16))
    prior: f32,
    n: u16,
    w: f32,
    proven: Proven,
    first_child: u32,    // index of this node's first child in the arena
    n_children: u16,     // 0 means "not expanded yet" (a leaf)
}

struct Tree {
    nodes: Vec<Node>,    // the arena; index 0 is the root
}

impl Tree {

}