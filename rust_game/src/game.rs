// this file was made with significant assistance from AI

use std::cmp::Ordering;
use once_cell::sync::Lazy;
use crate::game::GameResult::WhiteWin;

pub const N: usize = 6;
pub const STRIDE: usize = 8;
pub const SQUARES: usize = N * STRIDE;      // 48 slots, 12 are padding
pub const STONES: u8 = 30;                  // flats + walls, per color
pub const CAPS: u8 = 1;
pub const BOARD_MASK: u64 = 0x3F3F_3F3F_3F3F; // bits y*8+x for x,y in 0..6

/// Direction order matches the Python PTN symbols: '+', '-', '<', '>'.
pub const DIR_STEP: [isize; 4] = [8, -8, -1, 1];

// ---------------------------------------------------------------- basics

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Color { White = 0, Black = 1 }

impl Color {
    #[inline] pub fn idx(self) -> usize { self as usize }
    #[inline] pub fn other(self) -> Color {
        match self { Color::White => Color::Black, Color::Black => Color::White }
    }
}

/// Type of the TOP piece of a stack. Only the top can be a wall or capstone.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Kind { Flat = 0, Wall = 1, Cap = 2 }

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GameResult { Ongoing, WhiteWin, BlackWin, Draw }

// ---------------------------------------------------------------- stack

/// Piece colors packed 1 bit each: 1 = black, top piece at bit 0.
/// Bits at or above `height` must always be zero.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Stack {
    pub bits: u64,
    pub height: u8,   // max 63 on 6x6 (realistic max is about 60)
    pub kind: Kind,   // kind of the top piece; meaningless when height == 0
}
/*
potential alternative encoding scheme for Stack, the struct is compressed into a u64, the first 61 bits would be allocated to stack representation
there would be a sentinel bit at position equivilant to stack height, all bits under the sentitel would inidicate stack composition, 2 bits would be assigned to indicate kind
*/
impl Stack {
    pub const EMPTY: Stack = Stack { bits: 0, height: 0, kind: Kind::Flat };


    #[inline] pub fn is_empty(self) -> bool { self.height == 0 }

    #[inline] pub fn top_color(self) -> Color {
        debug_assert!(self.height > 0);
        if self.bits & 1 == 0 { Color::White } else { Color::Black }
    }

    /// Place one piece on top (placements only ever target empty squares).
    #[inline] pub fn push(&mut self, color: Color, kind: Kind) {
        debug_assert!(self.height < 63);
        debug_assert!(self.kind != Kind::Cap && self.kind != Kind::Wall);
        self.bits = (self.bits << 1) | color as u64;
        self.height += 1;
        self.kind = kind;
    }

    /// Remove the top `k` pieces from `self` and returns them as a `Stack` a flat always tops What remains.
    #[inline] pub fn lift(&mut self, k: u8) -> Stack {
        debug_assert!(k >= 1 && k <= self.height);
        let carried: Stack = Stack { bits: self.bits & ((1u64 << k) - 1), height: k, kind: self.kind };
        self.bits >>= k;
        self.height -= k;
        self.kind = Kind::Flat;
        carried
    }

    /// Put `chunk` of type `Stack` on top of `self`.
    /// replaces `self.kind` with `chunk.kind` and adds `chunk.height` to `self.height`
    #[inline] pub fn put(&mut self, chunk: Stack) {
        debug_assert!(self.height + chunk.height <= 63);
        self.bits = (self.bits << chunk.height) | chunk.bits;
        self.height += chunk.height;
        self.kind = chunk.kind;
    }

    #[inline] pub fn put_bottom(&mut self, chunk: Stack) {
        debug_assert!(self.height + chunk.height <= 63);
        if self.height == 0 {
            self.kind = chunk.kind;
        }
        self.bits = (chunk.bits << self.height) | self.bits;
        self.height += chunk.height;
    }

    /// remove the bottom `d` pieces from `self` and returns them as a `Stack`
    /// the `kind` of output will be `Kind::Flat` will reject the drop if the entire stack is dropped leaving height 0 behind
    #[inline] pub fn drop_bottom(&mut self, d: u8) -> Stack {
        debug_assert!(d > 0 && d <= self.height);
        let r = self.height - d;                       // pieces that stay
        let dropped = Stack { bits: self.bits >> r, height: d, kind: if r == 0 { self.kind } else { Kind::Flat} };
        self.bits &= (1u64 << r) - 1;
        self.height = r;
        dropped
    }
}

// ---------------------------------------------------------------- move

/// Packed u16.
///   bit 0      : 0 = placement, 1 = slide
///   bits 1..=6 : square index (y*8 + x)
/// Placement:
///   bits 7..=8 : Kind (0 flat, 1 wall, 2 cap)
/// Slide:
///   bits 7..=8 : direction (0 '+', 1 '-', 2 '<', 3 '>')
///   bits 9..=14: drop mask. Bit (c-1) is set for every cumulative drop
///                total c. Carry count = highest set bit + 1.
///                Example: carry 6 with drops 1,2,3 -> cumulative 1,3,6
///                -> bits 0, 2, 5 -> 0b100101.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Move(u16);

impl Move {
    #[inline] pub fn place(sq: usize, kind: Kind) -> Move {
        Move(((sq as u16) << 1) | ((kind as u16) << 7))
    }

    #[inline] pub fn slide(sq: usize, dir: u8, mask: u8) -> Move {
        debug_assert!(mask != 0 && mask < 63 && dir < 4);
        Move(1 | ((sq as u16) << 1) | ((dir as u16) << 7) | ((mask as u16) << 9))
    }

    #[inline] pub fn is_slide(self) -> bool { self.0 & 1 == 1 }
    #[inline] pub fn square(self) -> usize { ((self.0 >> 1) & 0x3F) as usize }

    #[inline] pub fn place_kind(self) -> Kind {
        match (self.0 >> 7) & 3 { 0 => Kind::Flat, 1 => Kind::Wall, _ => Kind::Cap }
    }

    #[inline] pub fn dir(self) -> u8 { ((self.0 >> 7) & 3) as u8 }
    #[inline] pub fn mask(self) -> u8 { ((self.0 >> 9) & 0x3F) as u8 }

    /// Total stones lifted from the origin stack.
    #[inline] pub fn carry(self) -> u8 { 8 - self.mask().leading_zeros() as u8 }

    /// Drop counts per square, in travel order. Returns (counts, how_many).
    #[inline] pub fn drops(self) -> ([u8; 5], usize) {
        let m = self.mask();
        let mut out = [0u8; 5];
        let (mut n, mut prev) = (0usize, 0u8);
        for i in 0..6u8 {
            if (m >> i) & 1 == 1 {
                out[n] = (i + 1) - prev;
                prev = i + 1;
                n += 1;
            }
        }
        (out, n)
    }

    /// Returns a PNT formatted `String` which represents `self`
    pub fn to_ptn(self) -> String {
        let sq = self.square();
        let file = (b'a' + (sq & 7) as u8) as char;
        let rank = (b'1' + (sq >> 3) as u8) as char;

        if !self.is_slide() {
            let prefix = match self.place_kind() {
                Kind::Flat => "",
                Kind::Wall => "S",
                Kind::Cap => "C",
            };
            return format!("{prefix}{file}{rank}");
        }

        let dir = match self.dir() { 0 => '+', 1 => '-', 2 => '<', _ => '>' };
        let carry = self.carry();
        let (d, n) = self.drops();

        let mut out = String::new();
        if carry > 1 {
            out.push_str(&carry.to_string());
        }
        out.push(file);
        out.push(rank);
        out.push(dir);
        if n > 1 {
            for &dropped in &d[..n] {
                out.push_str(&dropped.to_string());
            }
        }
        out
    }

    /// Accepts a PNT formatted `String` and returns a `Move`
    pub fn from_ptn(ptn_move: &str) -> Move {
        let mut direction: i8 = -1;
        if ptn_move.contains('+') {
            direction = 0
        } else if ptn_move.contains('-') {
            direction = 1
        } else if ptn_move.contains('<') {
            direction = 2
        } else if ptn_move.contains('>') {
            direction = 3
        }
        if direction == -1 {
            let kind = match ptn_move.as_bytes()[0] {
                b'S' => Kind::Wall,
                b'C' => Kind::Cap,
                _ => Kind::Flat,
            };
            if kind == Kind::Flat {
                let file = (ptn_move.as_bytes()[0] - b'a') as usize;
                let rank = (ptn_move.as_bytes()[1] - b'1') as usize;
                let sq = rank * 8 + file;
                return Move::place(sq, kind);
            }
            let file = (ptn_move.as_bytes()[1] - b'a') as usize;
            let rank = (ptn_move.as_bytes()[2] - b'1') as usize;
            let sq = rank * 8 + file;
            return Move::place(sq, kind);
        }
        let mut carry = match ptn_move.as_bytes()[0] {
            b'2' => 2,
            b'3' => 3,
            b'4' => 4,
            b'5' => 5,
            b'6' => 6,
            b'1' => 1,
            _ => 0, // the carry here is actually 1, but I need to be able to distinguish it from the explisit 1 case
        };
        if carry == 0 {
            let file = (ptn_move.as_bytes()[0] - b'a') as usize;
            let rank = (ptn_move.as_bytes()[1] - b'1') as usize;
            let sq = rank * 8 + file;
            return Move::slide(sq, direction as u8, 1u8);
        }
        let mut sequence: Vec<u8> = Vec::new();
        let seq_len = ptn_move.len() - 4;
        if seq_len > 0 {
            for i in 0..seq_len {
                sequence.push(ptn_move.as_bytes()[i + 4] - b'0');
            }
        } else {
            sequence.push(carry)
        }
        let mut move_mask: u8 = 0;
        let mut running_sum: u8 = 0;
        for amount in sequence {
            running_sum += amount;
            move_mask |= (1u8 << (running_sum - 1));
        }
        let file = (ptn_move.as_bytes()[1] - b'a') as usize;
        let rank = (ptn_move.as_bytes()[2] - b'1') as usize;
        let sq = rank * 8 + file;
        Move::slide(sq, direction as u8, move_mask)
    }

    /// Bitboard of every square this move changes (origin plus landing squares).
    #[inline] pub fn touched(self) -> u64 {
        let origin = self.square();
        let mut bits = 1u64 << origin;
        if self.is_slide() {
            let step = DIR_STEP[self.dir() as usize];
            let (_, n) = self.drops();
            let mut sq = origin as isize;
            for _ in 0..n {
                sq += step;
                bits |= 1u64 << sq;
            }
        }
        bits
    }
}

#[inline] fn set_bit(bitboard: &mut u64, sq: usize) {
    *bitboard |= 1u64 << sq;
}

#[inline] fn clear_bit(bitboard: &mut u64, sq: usize) {
    *bitboard &= !(1u64 << sq);
}

#[inline] fn flip_bit(bitboard: &mut u64, sq: usize) {
    *bitboard ^= 1u64 << sq;
}

#[inline] fn write_bit(bitboard: &mut u64, sq: usize, val: u8) {
    if val == 0 {
        clear_bit(bitboard, sq);
    } else {
        set_bit(bitboard, sq);
    }
}

#[inline] fn shift_bit(bitboard: &u64, shift: isize) -> u64 {
    if shift >= 0 { bitboard << shift } else { bitboard >> (-shift) }
}

#[inline] fn get_bit(bitboard: u64, sq: usize) -> u8 {
    ((bitboard >> sq) & 1) as u8
}



// ---------------------------------------------------------------- undo

#[derive(Clone, Copy, Debug)]
pub struct Undo {
    pub mv: Move,
    /// A capstone's final drop flattened a wall (restore Wall on undo).
    pub flattened: bool,
}

// ------------- stuff needed for road detection -----------------------

const LEFT_EDGE: u64 = 0x0101_0101_0101;     // bits 0, 8, ..., 40
const RIGHT_EDGE: u64 = LEFT_EDGE << 5;
const BOTTOM_EDGE: u64 = 0x3F;
const TOP_EDGE: u64 = 0x3F << 40;

/// Connected component of `road` containing `seed` (seed must be a subset of road).
/// No column masks are needed: shifts that cross a row land on padding bits,
/// which are never set in `road`.
#[inline] fn flood(seed: u64, road: u64) -> u64 {
    let mut f = seed;
    loop {
        let next = (f | (f << 1) | (f >> 1) | (f << 8) | (f >> 8)) & road;
        if next == f { return f; }
        f = next;
    }
}

#[inline] fn spans(comp: u64) -> bool {
    ((comp & LEFT_EDGE) != 0 && (comp & RIGHT_EDGE) != 0)
        || ((comp & BOTTOM_EDGE) != 0 && (comp & TOP_EDGE) != 0)
}
const CARRY_OPTIONS: u8 = 6; // values 1..=6 are valid
const LENGTH_OPTIONS: u8 = 6; // values 0..6 are valid
const SOFT_END: u8 = 2; // values 0 and 1 are valid, effectivly this is a bianary flip
const ARRAY_LENGTH: u8 = CARRY_OPTIONS * LENGTH_OPTIONS * SOFT_END; // 72, note that only about half of these are valid

/// index will be computed to get a uniqe value from the following
fn get_mask_index(carry_capacity: &u8, drop_length: &u8, soft_end: &u8) -> usize {
    ((carry_capacity - 1) + (drop_length * 6) + (soft_end * 36)) as usize
}


static DROP_MASKS: Lazy<[Vec<u8>; ARRAY_LENGTH as usize]> = Lazy::new(|| {
    let mut masks = [const { Vec::<u8>::new() }; ARRAY_LENGTH as usize];
    assert_eq!(masks.len(), 72);
    for carry in 1..=CARRY_OPTIONS {
        for length in 0..LENGTH_OPTIONS {
            for soft in 0..SOFT_END {
                if length > carry {
                    continue;
                }
                if length == 5 && soft == 1 {
                    continue;
                }
                for mask in 1..(1u8 << carry) {
                    let p = mask.count_ones() as u8;
                    let c = 8 - mask.leading_zeros() as u8;               // this mask's own carry
                    let last_is_one = c == 1 || (mask >> (c - 2)) & 1 == 1;
                    if p <= length || (soft == 1 && p == length + 1 && last_is_one) {
                        masks[get_mask_index(&carry, &length, &soft)].push(mask);
                    }
                }
            }
        }
    }
    masks
});

// ---------------------------------------------------------------- game




#[derive(Clone, Debug)]
pub struct Tak {
    // ground truth
    pub stacks: [Stack; SQUARES],

    // derived bitboards, TOP pieces only
    pub top: [u64; 2],    // owner of the top piece, any type
    pub walls: u64,       // top is a wall
    pub caps: u64,        // top is a capstone

    // reserves
    pub stones_left: [u8; 2],
    pub caps_left: [u8; 2],

    // game state
    pub ply: u16,
    pub result: GameResult,

    // for unmake
    pub history: Vec<Undo>,
}

impl Tak {
    pub fn new() -> Tak {
        Tak {
            stacks: [Stack::EMPTY; SQUARES],
            top: [0; 2],
            walls: 0,
            caps: 0,
            stones_left: [STONES; 2],
            caps_left: [CAPS; 2],
            ply: 0,
            result: GameResult::Ongoing,
            history: Vec::new(),
        }
    }

    // ---- turn state -------------------------------------------------

    #[inline] pub fn side_to_move(&self) -> Color {
        if self.ply & 1 == 0 { Color::White } else { Color::Black }
    }

    /// First two plies: each side places a flat of the OPPONENT's color.
    #[inline] pub fn in_opening(&self) -> bool { self.ply < 2 }

    /// Color of the piece a placement puts down this ply.
    #[inline] pub fn placing_color(&self) -> Color {
        if self.in_opening() { self.side_to_move().other() } else { self.side_to_move() }
    }

    /// TPS/PTN move number.
    #[inline] pub fn move_number(&self) -> u16 { self.ply / 2 + 1 }

    // ---- derived boards (all one or two ops) -------------------------

    #[inline] pub fn occupied(&self) -> u64 { self.top[0] | self.top[1] }
    #[inline] pub fn empty(&self) -> u64 { !self.occupied() & BOARD_MASK }

    /// Squares that count for roads: flat or capstone tops.
    #[inline] pub fn road(&self, c: Color) -> u64 { self.top[c.idx()] & !self.walls }

    /// Squares that count for flat-win scoring: flat tops only.
    #[inline] pub fn flat_tops(&self, c: Color) -> u64 {
        self.top[c.idx()] & !self.walls & !self.caps
    }

    // ---- bitboard maintenance ---------------------------------------

    /// Recompute this square's bits in top/walls/caps from the stack.
    /// Call for every square a move touches, after changing its stack.
    #[inline] pub fn refresh_square(&mut self, sq: usize) {
        let bit = 1u64 << sq;
        self.top[0] &= !bit;
        self.top[1] &= !bit;
        self.walls &= !bit;
        self.caps &= !bit;
        let s = self.stacks[sq];
        if s.height == 0 { return; }
        self.top[s.top_color().idx()] |= bit;
        match s.kind {
            Kind::Wall => self.walls |= bit,
            Kind::Cap => self.caps |= bit,
            Kind::Flat => {}
        }
    }

    // ---- debugging --------------------------------------------------

    /// Recompute everything from `stacks` and assert it matches.
    /// Call under debug_assert after every make/unmake and from_tps.
    pub fn check_invariants(&self) {
        let (mut top, mut walls, mut caps) = ([0u64; 2], 0u64, 0u64);
        let mut on_board = [0u8; 2];
        for sq in 0..SQUARES {
            let s = self.stacks[sq];
            if (BOARD_MASK >> sq) & 1 == 0 {
                assert!(s.is_empty(), "padding square {sq} is occupied");
                continue;
            }
            assert!(s.height <= 63, "stack too tall at {sq}");
            if s.height == 0 {
                assert!(s.bits == 0);
                continue;
            }
            assert!(s.bits >> s.height == 0, "stray bits above height at {sq}");
            let bit = 1u64 << sq;
            top[s.top_color().idx()] |= bit;
            match s.kind {
                Kind::Wall => walls |= bit,
                Kind::Cap => caps |= bit,
                Kind::Flat => {}
            }
            let black = s.bits.count_ones() as u8;
            on_board[Color::Black.idx()] += black;
            on_board[Color::White.idx()] += s.height - black;
        }
        assert_eq!(top, self.top);
        assert_eq!(walls, self.walls);
        assert_eq!(caps, self.caps);
        for c in 0..2 {
            assert_eq!(
                on_board[c] + self.stones_left[c] + self.caps_left[c],
                STONES + CAPS,
                "piece count mismatch for color {c}"
            );
        }
    }

    fn handle_board_change(&mut self, mv: Move) {
        let origin = mv.square();
        let mut flattened = false;

        if mv.is_slide() {
            debug_assert!(self.stacks[origin].top_color() == self.side_to_move());
            debug_assert!(mv.carry() <= self.stacks[origin].height);

            let step = DIR_STEP[mv.dir() as usize];
            let (drops, n) = mv.drops();
            let mut carried = self.stacks[origin].lift(mv.carry());
            self.refresh_square(origin);

            let mut sq = origin as isize;
            for i in 0..n {
                sq += step;
                let target = sq as usize;
                if i == n - 1 {
                    let t = self.stacks[target];
                    flattened = t.height > 0 && t.kind == Kind::Wall;
                    debug_assert!(!flattened || (carried.kind == Kind::Cap && carried.height == 1));
                    debug_assert!(t.height == 0 || t.kind != Kind::Cap);
                    self.stacks[target].put(carried);
                } else {
                    let chunk = carried.drop_bottom(drops[i]);
                    self.stacks[target].put(chunk);
                }
                self.refresh_square(target);
            }
        } else {
            debug_assert!(self.stacks[origin].is_empty());
            let color = self.placing_color();
            let kind = mv.place_kind();
            self.stacks[origin].push(color, kind);
            if kind == Kind::Cap {
                self.caps_left[color.idx()] -= 1;
            } else {
                self.stones_left[color.idx()] -= 1;
            }
            self.refresh_square(origin);
        }

        self.history.push(Undo { mv, flattened });

    }



    fn color_has_new_road(&self, c: Color, touched: u64) -> bool {
        let road = self.road(c);
        if road.count_ones() < 6 { return false; }
        let mut seeds = touched & road;
        while seeds != 0 {
            let seed = seeds & seeds.wrapping_neg();   // lowest set bit
            let comp = flood(seed, road);
            if spans(comp) { return true; }
            seeds &= !comp;                            // skip seeds in the same component
        }
        false
    }

    /// Sets self.result and returns true if the last move created a road.
    /// Must run before ply is incremented, so side_to_move() is the mover.
    fn identify_road_win(&mut self) -> bool {
        let last_move_index = self.history.len() - 1;
        let touched = self.history[last_move_index].mv.touched();
        let mover = self.side_to_move();
        // The mover wins if both players complete a road, so check the mover first.
        for c in [mover, mover.other()] {
            if self.color_has_new_road(c, touched) {
                self.result = match c {
                    Color::White => GameResult::WhiteWin,
                    Color::Black => GameResult::BlackWin,
                };
                return true;
            }
        }
        false
    }

    #[inline] fn identify_flat_win(&self) -> bool {
        (((self.top[0] | self.top[1]) & BOARD_MASK) == BOARD_MASK) // all spaces on board are occupied
        || (self.caps_left[0] + self.stones_left[0] == 0) // white has no peices left
            || (self.caps_left[1] + self.stones_left[1] == 0) // black has no peices left
    }


    fn find_result(&mut self) { // the method for checking for a win or loss, or if the game is ongoing
        if self.ply >= 10 { // the board is 6 spaces across, so 5 is the first ply where a road win is possible
            if self.identify_road_win() {
                return // road wins take priority over flat wins, so if a road win happened, a flat check is redundant
            }
        }
        if self.ply >= 35 { // there are 36 spaces on the board, so 35 is the first ply where a flat win is possible
            if self.identify_flat_win() {
                self.result = match self.flat_tops(Color::White).count_ones().cmp(&self.flat_tops(Color::Black).count_ones()) {
                    Ordering::Equal => GameResult::Draw,
                    Ordering::Greater => GameResult::WhiteWin,
                    Ordering::Less => GameResult::BlackWin,
                }
            }
        }
    }

    pub fn make_move(&mut self, board_move: Move) {
        self.handle_board_change(board_move);
        self.find_result();
        self.ply += 1;
    }


    fn undo_board_change(&mut self, mv: Move, flattened: bool) {
        let origin = mv.square();
        if mv.is_slide() {
            let step = -DIR_STEP[mv.dir() as usize];
            let (drops, n) = mv.drops();
            let mut carried = Stack::EMPTY;
            for i in (1..=n).rev() {
                let location = (origin as isize - step * i as isize) as usize;
                let chunk = self.stacks[location].lift(drops[i - 1]);
                carried.put_bottom(chunk);
                if i == n && flattened {
                    self.stacks[location].kind = Kind::Wall;
                }
                self.refresh_square(location);
            }
            self.stacks[origin].put(carried);
            self.refresh_square(origin);
        } else {
            let color = self.stacks[origin].top_color();
            self.stacks[origin] = Stack::EMPTY;
            if mv.place_kind() == Kind::Cap {
                self.caps_left[color.idx()] += 1;
            } else {
                self.stones_left[color.idx()] += 1;
            }
            self.refresh_square(origin);
        }
    }

    pub fn unmake_move(&mut self) {
        let undo = self.history.pop().unwrap();
        self.ply -= 1;
        self.undo_board_change(undo.mv, undo.flattened);
        self.result = GameResult::Ongoing;
    }

    #[inline] fn get_move_distance(&self, sq: usize, step: isize, border: u64, max_steps: usize) -> (usize, u8) {
        let mut point: u64 = 1u64 << sq;
        for i in 0..max_steps {
            if point & border != 0 {
                return (i, 0)
            }
            point = shift_bit(&point, step);
            if point & self.caps != 0 {
                return (i, 0)
            }
            if point & self.walls != 0 {
                return (i, 1)
            }
        }
        (max_steps, 0)
    }


    pub fn get_all_legal_moves(&self, output: &mut Vec<Move>) {
        debug_assert_eq!(self.result, GameResult::Ongoing);
        output.clear();
        let side = self.side_to_move();
        let stones_remain = self.stones_left[side.idx()] > 0;
        let caps_remain = self.caps_left[side.idx()] > 0;
        for y in 0..6 {
            for x in 0..6 {
                let square: usize = y * 8 + x;
                let stack: Stack = self.stacks[square];
                if stack.height == 0 { // the space is empty
                    if stones_remain {
                        output.push(Move::place(square, Kind::Flat));
                        output.push(Move::place(square, Kind::Wall));
                    }
                    if caps_remain {
                        output.push(Move::place(square, Kind::Cap));
                    }
                } else {
                    if stack.top_color() == side && !self.in_opening() {
                        let (up_dist, soft_up) = self.get_move_distance(square, 8, TOP_EDGE, stack.height as usize);
                        let (down_dist, soft_down) = self.get_move_distance(square, -8, BOTTOM_EDGE, stack.height as usize);
                        let (left_dist, soft_left) = self.get_move_distance(square, -1, LEFT_EDGE, stack.height as usize);
                        let (right_dist, soft_right) = self.get_move_distance(square, 1, RIGHT_EDGE, stack.height as usize);
                        let has_cap_top = if stack.kind == Kind::Cap { 1u8 } else { 0u8 };
                        let move_capacity = if stack.height > 6 { 6 } else { stack.height };
                        let up_masks = &DROP_MASKS[get_mask_index(&move_capacity, &(up_dist as u8), &(soft_up & has_cap_top))];
                        let down_masks = &DROP_MASKS[get_mask_index(&move_capacity, &(down_dist as u8), &(soft_down & has_cap_top))];
                        let left_masks = &DROP_MASKS[get_mask_index(&move_capacity, &(left_dist as u8), &(soft_left & has_cap_top))];
                        let right_masks = &DROP_MASKS[get_mask_index(&move_capacity, &(right_dist as u8), &(soft_right & has_cap_top))];

                        for mask in up_masks {
                            output.push(Move::slide(square, 0, *mask));
                        }
                        for mask in down_masks {
                            output.push(Move::slide(square, 1, *mask));
                        }
                        for mask in left_masks {
                            output.push(Move::slide(square, 2, *mask));
                        }
                        for mask in right_masks {
                            output.push(Move::slide(square, 3, *mask));
                        }
                    }
                }
            }
        }
    }

    pub fn get_filtered_moves(&mut self, depth: u8, do_pruning: bool) -> (Vec<Move>, i8) {
        debug_assert_eq!(self.result, GameResult::Ongoing);
        let mut best_moves: Vec<Move> = Vec::new();
        let mut best_score: i8 = -1;
        let mut allocated_memory: Vec<Vec<Move>> = Vec::new();
        for _ in 0..=depth {
            allocated_memory.push(Vec::new());
        }
        self.get_all_legal_moves(&mut allocated_memory[0]);
        let (layer0, remaining_layers) = allocated_memory.split_at_mut(1);
        let me = self.side_to_move();
        for mv in &layer0[0] {
            self.make_move(*mv);
            let score: i8 = match self.result {
                GameResult::Ongoing if depth == 0 => 0,
                GameResult::Ongoing => -self.filter_recursive(depth - 1, do_pruning, remaining_layers),
                GameResult::WhiteWin => if me == Color::White { 1 } else { -1 },
                GameResult::BlackWin => if me == Color::Black { 1 } else { -1 } ,
                GameResult::Draw => 0,
            };
            self.unmake_move();
            if score == best_score {
                best_moves.push(*mv);
            } else if score > best_score {
                best_score = score;
                best_moves.clear();
                best_moves.push(*mv);
            }
            if do_pruning && best_score == 1 {
                break;
            }
        }
        (best_moves, best_score)
    }


    fn filter_recursive(&mut self, depth: u8, do_pruning: bool, allocated_memory: &mut [Vec<Move>]) -> i8 {
        self.get_all_legal_moves(&mut allocated_memory[0]);
        let mut score: i8 = -1;
        let (layer0, remaining_layers) = allocated_memory.split_at_mut(1);
        let me = self.side_to_move();
        for mv in &layer0[0] {
            self.make_move(*mv);
            let found_score = match self.result {
                GameResult::Ongoing if depth == 0 => 0,
                GameResult::Ongoing => -self.filter_recursive(depth - 1, do_pruning, remaining_layers),
                GameResult::WhiteWin => if me == Color::White { 1 } else { -1 } ,
                GameResult::BlackWin => if me == Color::Black { 1 } else { -1 },
                GameResult::Draw => 0,
            };
            self.unmake_move();
            if found_score > score {
                score = found_score;
            }
            if found_score == 1 {
                break;
            }
        }
        score
    }

    pub fn from_tps(tps: &str) -> Tak {
        Tak::new()
    }

    pub fn to_tps(&self) -> String {
        String::new()
    }

    /// convert the board state to a tensor that can be used for machine learning, tensor output is as follows,
    pub fn to_tensor(&self) -> String {
        String::new() // I will change it to something other than a string later
    }
}

impl PartialEq for Tak {
    fn eq(&self, other: &Self) -> bool {
        self.stacks == other.stacks
            && self.stones_left == other.stones_left
            && self.caps_left == other.caps_left
            && self.ply == other.ply
            && self.result == other.result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stack_behaves_as_expected() {
        let mut stack = Stack::EMPTY;
        assert!(stack.is_empty());
        stack.push(Color::White, Kind::Flat);
        assert_eq!(stack.height, 1);
        assert_eq!(stack.kind, Kind::Flat);
        assert_eq!(stack.top_color(), Color::White);
        stack.push(Color::Black, Kind::Wall);
        assert_eq!(stack.kind, Kind::Wall);
        assert_eq!(stack.height, 2);
        assert_eq!(stack.top_color(), Color::Black);

        let mut top_stone = stack.lift(1);
        assert_eq!(top_stone.top_color(), Color::Black);
        assert_eq!(top_stone.height, 1);

        assert_eq!(stack.top_color(), Color::White);
        assert_eq!(stack.height, 1);
        stack.put(top_stone);
        assert_eq!(stack.top_color(), Color::Black);
        assert_eq!(stack.height, 2);
        assert_eq!(stack.kind, Kind::Wall);
        let mut stack_bottom = stack.drop_bottom(1);
        assert_eq!(stack_bottom.top_color(), Color::White);
        assert_eq!(stack_bottom.height, 1);

        assert_eq!(stack.top_color(), Color::Black);
        assert_eq!(stack.kind, Kind::Wall);
        assert_eq!(stack.height, 1);
        stack_bottom.put(stack);
        assert_eq!(stack_bottom.height, 2);
        let dropped_parts = stack_bottom.drop_bottom(2);
        assert_eq!(dropped_parts.height, 2);
        assert_eq!(stack_bottom.height, 0);
    }

    #[test]
    fn move_behaves_as_expected() {
        let place_move = Move::place(0, Kind::Flat);
        assert_eq!(place_move.place_kind(), Kind::Flat);
        assert_eq!(place_move.is_slide(), false);
        let slide_move = Move::slide(0, 0, 1);
        assert_eq!(slide_move.is_slide(), true);
    }

    #[test]
    fn ptn_conversion_words() {
        let move_1 = String::from("c5");
        let move_2 = String::from("Ca2");
        let move_3 = String::from("Sf6");

        let move_4 = String::from("d5+");
        let move_5 = String::from("6a3-");
        let move_6 = String::from("4d5>211");
        let move_7 = String::from("6e2<2112");

        println!("move_1: {}", move_1);

        let converted_1 = Move::from_ptn(&move_1);
        let converted_2 = Move::from_ptn(&move_2);
        let converted_3 = Move::from_ptn(&move_3);

        let converted_4 = Move::from_ptn(&move_4);
        let converted_5 = Move::from_ptn(&move_5);
        let converted_6 = Move::from_ptn(&move_6);
        let converted_7 = Move::from_ptn(&move_7);

        assert_eq!(converted_1.is_slide(), false);
        assert_eq!(converted_2.is_slide(), false);
        assert_eq!(converted_3.is_slide(), false);
        assert_eq!(converted_1.place_kind(), Kind::Flat);
        assert_eq!(converted_2.place_kind(), Kind::Cap);
        assert_eq!(converted_3.place_kind(), Kind::Wall);

        assert_eq!(converted_4.is_slide(), true);
        assert_eq!(converted_5.is_slide(), true);
        assert_eq!(converted_6.is_slide(), true);
        assert_eq!(converted_7.is_slide(), true);

        assert_eq!(converted_1.to_ptn(), move_1);
        assert_eq!(converted_2.to_ptn(), move_2);
        assert_eq!(converted_3.to_ptn(), move_3);
        assert_eq!(converted_4.to_ptn(), move_4);
        assert_eq!(converted_5.to_ptn(), move_5);
        assert_eq!(converted_6.to_ptn(), move_6);
        assert_eq!(converted_7.to_ptn(), move_7);
    }

    #[test]
    fn tak_methods_dont_crash() {
        let mut tak = Tak::new();
        assert_eq!(tak.move_number(), 1);
    }

    #[test]
    fn tak_board_state_changes_work() {
        let move_sequence = [
            "a6", "f6",
            "d4", "c4",
            "d3", "c3",
            "d5", "Ce3",
            "d6", "c2",
            "Cc5", "d2",
            "b4", "e4",
            "e6", "a4",
            "b5", "a5",
            "b3", "1c4<1",
            "c4", "2b4-2",
            "1c5<1", "1e4<1",
            "c5", "2d4+2",
            "2b5-11", "e2",
            "c6", "1c3+1",
            "a3", "2c4+2",
            "b5", "b6",
            "1b5>1", "3d5<3",
            "1c6-1", "Sc4",
            "6c5<33", "1c4+1",
            "d4", "1a4+1",
            "1b5<1", "1a6-1",
            "Sa4", "6a5+6",
            "Sc6", "d5",
            "e5", "f3",
            "c4", "1d5-1",
            "1d3+1", "3c5-3",
            "3b3>12", "4c4-4",
            "1d3<1", "1a5>1",
            "4c3+4", "3b5-12",
            "d5", "1c2+1",
            "2c4-2", "1d3+1",
            "1c4>1", "2b4>11",
            "1c4>1", "a1",
            "4c3-13",
        ];
        let mut tak = Tak::new();
        let mut forward = vec![tak.clone()];          // forward[k] = state after k moves
        for ptn in &move_sequence {
            tak.make_move(Move::from_ptn(ptn));
            forward.push(tak.clone());
        }
        for k in (0..move_sequence.len()).rev() {
            println!("unmaking move {} {}", k + 1, move_sequence[k]);
            tak.unmake_move();
            assert_eq!(tak, forward[k], "mismatch after unmaking move {}", k + 1);
        }
    }

    #[test]
    fn tak_win_detection_works() {
        let black_win_test  = [
            "e6", "a1",
            "b1", "e5",
            "a2", "e4",
            "b2", "e3",
            "a3", "e2",
            "b3", "Cf6",
            "a4", "Sf5",
            "b4", "Sf4",
            "a5", "Sf3",
            "b5", "f6<",
            "c1", "Sa6",
            "c2", "Sb6",
            "c3", "Sc6",
            "c4", "Sd6",
            "c5", "Sf6",
            "d5", "Sf2",
            "d1", "d4",
            "d1<", "d1",
            "2c1>", "Sc1",
            "d2", "Sd3",
            "3d1+", "f1",
            "4d2-", "Sd2",
        ];
        let mut tak = Tak::new();
        for ptn in &black_win_test {
            tak.make_move(Move::from_ptn(ptn));
            assert_eq!(tak.result, GameResult::Ongoing);
        }
        tak.make_move(Move::from_ptn("3d1>12"));
        assert_eq!(tak.result, GameResult::BlackWin);

        let dual_road_white_win = [
            "f6", "a1",
            "a2", "f5",
            "a3", "f4",
            "a4", "b1",
            "a5", "b2",
            "c1", "b3",
            "c2", "b4",
            "c3", "b5",
            "c4", "d2",
            "d6", "c6",
            "d6<", "f3",
        ];

        let mut tak = Tak::new();
        for ptn in &dual_road_white_win {
            tak.make_move(Move::from_ptn(ptn));
            assert_eq!(tak.result, GameResult::Ongoing);
        }
        tak.make_move(Move::from_ptn("2c6<11"));
        assert_eq!(tak.result, GameResult::WhiteWin);

        let running_out_of_stones = [
            "b6", "e6",
            "Cc1", "e4",
            "c5", "Ce1",
            "d6", "a1",
            "Sd5", "d2",
            "f1", "f6",
            "d4", "a6",
            "d4>", "Sc3",
            "f4", "e1>",
            "b2", "Sc2",
            "e6>", "a3",
            "d1", "Se2",
            "b4", "Sb1",
            "d1+", "d3",
            "c1+", "a4",
            "e5", "c1",
            "f2", "a2",
            "a5", "Sd1",
            "Se3", "d3-",
            "d4", "d3",
            "b2<", "f3",
            "a5-", "f3+",
            "2a2-", "3d2+12",
            "b3", "a3>",
            "f3", "a5",
            "2c2-", "b2",
            "2a4+11", "2a5>11",
            "c6", "e2>",
            "d5-", "f5",
            "Sa5", "2f4-",
            "3d4-12", "b6<",
            "Sa3", "3a6>111",
            "c4", "a4",
            "c6-", "f5<",
            "2c5<", "a4>",
            "Se6", "2d6<",
            "c4<", "a6",
            "3b4-12", "e1",
            "3b5+", "3c6-",
            "3b3+12", "4c5<",
            "Sa2", "6b5-51",
            "a4", "b3+",
            "Sf4", "c3>",
            "2e4+", "4d3+13",
            "Sc2", "Sc3",
            "e2", "5b4<",
            "Sd6", "3d5>12",
            "2b6<", "2f5+",
            "f4-", "2b6-",
            "a3+", "4f6-13",
            "5a4>1112", "a3",
            "b3", "2b5-11",
        ];

        let mut tak = Tak::new();
        for ptn in &running_out_of_stones {
            tak.make_move(Move::from_ptn(ptn));
            assert_eq!(tak.result, GameResult::Ongoing);
        }
        tak.make_move(Move::from_ptn("d3"));
        assert_eq!(tak.result, GameResult::WhiteWin);

        let board_space_used_up = [
            "f1", "f2",
            "f3", "f4",
            "f5", "f6",
            "e6", "Se5",
            "e4", "e3",
            "e2", "e1",
            "d1", "d2",
            "d3", "d4",
            "d5", "d6",
            "d5-", "Cd5",
            "c6", "c5",
            "c6>", "c6",
            "2d6>", "d6",
            "2d4-", "d4",
            "c4", "c3",
            "c2", "c1",
            "b1", "b2",
            "b3", "b4",
            "b5", "b6",
            "a6", "a5",
            "a4", "a3",
            "a2", "c3<",
            "a1",
        ];
        let mut tak = Tak::new();
        for ptn in &board_space_used_up {
            tak.make_move(Move::from_ptn(ptn));
            assert_eq!(tak.result, GameResult::Ongoing);
        }
        tak.make_move(Move::from_ptn("b3>"));
        assert_eq!(tak.result, GameResult::Draw);
    }

    #[test]
    fn test_getting_legal_moves() {
        let mut tak = Tak::new();
        let mut i: u32 = 0;
        let mut legal_moves: Vec<Move> = Vec::new();
        loop {
            tak.get_all_legal_moves(&mut legal_moves);
            let num_moves = legal_moves.len();
            assert!(num_moves > 0);
            for move_ in &legal_moves {
                tak.make_move(*move_);
                tak.check_invariants();
                tak.unmake_move();
                tak.check_invariants();
            }
            let selected_move = legal_moves[i as usize % num_moves];
            println!("move {}: {}", i, selected_move.to_ptn());
            tak.make_move(selected_move);
            i += 1;
            if tak.result != GameResult::Ongoing {
                break;
            }
        }
        let mut tak = Tak::new();
        let move_sequence = [
            "b6", "e6",
            "Cc1", "e4",
            "c5", "Ce1",
            "d6", "a1",
            "Sd5", "d2",
            "f1", "f6",
            "d4", "a6",
            "d4>", "Sc3",
            "f4", "e1>",
            "b2", "Sc2",
            "e6>", "a3",
            "d1", "Se2",
            "b4", "Sb1",
            "d1+", "d3",
            "c1+", "a4",
            "e5", "c1",
            "f2", "a2",
            "a5", "Sd1",
            "Se3", "d3-",
            "d4", "d3",
            "b2<", "f3",
            "a5-", "f3+",
            "2a2-", "3d2+12",
            "b3", "a3>",
            "f3", "a5",
            "2c2-", "b2",
            "2a4+11", "2a5>11",
            "c6", "e2>",
            "d5-", "f5",
            "Sa5", "2f4-",
            "3d4-12", "b6<",
            "Sa3", "3a6>111",
            "c4", "a4",
            "c6-", "f5<",
            "2c5<", "a4>",
            "Se6", "2d6<",
            "c4<", "a6",
            "3b4-12", "e1",
            "3b5+", "3c6-",
            "3b3+12", "4c5<",
            "Sa2", "6b5-51",
            "a4", "b3+",
            "Sf4", "c3>",
            "2e4+", "4d3+13",
            "Sc2", "Sc3",
            "e2", "5b4<",
            "Sd6", "3d5>12",
            "2b6<", "2f5+",
            "f4-", "2b6-",
            "a3+", "4f6-13",
            "5a4>1112", "a3",
            "b3", "2b5-11",
        ];
        for ptn in &move_sequence {
            tak.get_all_legal_moves(&mut legal_moves);
            assert!(legal_moves.contains(&Move::from_ptn(ptn)));
            tak.make_move(Move::from_ptn(ptn));
        }
        let move_sequence = [
            "b6", "e6",
            "Cc1", "e4",
            "c5", "Ce1",
            "d6", "a1",
            "Sd5", "d2",
            "f1", "f6",
            "d4", "a6",
            "d4>", "Sc3",
            "f4", "e1>",
            "b2", "Sc2",
            "e6>", "a3",
            "d1", "Se2",
            "b4", "Sb1",
            "d1+", "d3",
            "c1+*", "a4",
            "e5", "c1",
            "f2", "a2",
            "a5", "Sd1",
            "Se3", "d3-",
            "d4", "d3",
            "b2<", "f3",
            "a5-", "f3+",
            "2a2-", "3d2+12",
            "b3", "a3>",
            "f3", "a5",
            "2c2-", "b2",
            "2a4+11", "2a5>11",
            "c6", "e2>",
            "d5-", "f5",
            "Sa5", "2f4-",
            "3d4-12", "b6<",
            "Sa3", "3a6>111",
            "c4", "a4",
            "c6-", "f5<",
            "2c5<", "a4>",
            "Se6", "2d6<",
            "c4<", "a6",
            "3b4-12", "e1",
            "3b5+", "3c6-",
            "3b3+12", "4c5<",
            "Sa2", "6b5-51",
            "a4", "b3+",
            "Sf4", "c3>",
            "2e4+", "4d3+13",
            "Sc2", "Sc3",
            "e2", "5b4<",
            "Sd6", "3d5>12",
            "2b6<", "2f5+",
            "f4-", "2b6-",
            "a3+", "4f6-13",
            "5a4>1112", "a3",
            "b3",
        ];
        let legal_moves_known = [
            "1b1+1", "1b1<1", "1d1>1", "1e1+1", "1f1+1", "1f1<1", "2f1<11", "2f1<2", "1f2<1",
            "2f2<2", "1a3+1", "1a3>1", "1c3+1", "1c3<1", "1c3>1", "d3", "Sd3", "1f4+1", "2f4+11",
            "3f4+12", "2f4+2", "3f4+21", "3f4+3", "1b5-1", "2b5-11", "2b5-2", "1b5+1", "2b5+2",
            "1b5>1", "2b5>11", "2b5>2", "c5", "Sc5", "d5", "Sd5", "1e5<1", "2e5<11", "3e5<111",
            "4e5<112", "5e5<113", "3e5<12", "4e5<121", "5e5<122", "4e5<13", "5e5<131", "5e5<14",
            "2e5<2", "3e5<21", "4e5<211", "5e5<212", "4e5<22", "5e5<221", "5e5<23", "3e5<3", "4e5<31",
            "5e5<311", "5e5<32", "4e5<4", "5e5<41", "5e5<5", "1e5>1", "2e5>2", "3e5>3", "4e5>4", "5e5>5",
            "1f5+1", "1f5<1", "b6", "Sb6", "c6", "Sc6", "f6", "Sf6"
        ];
        let legal_moves_known = legal_moves_known.iter().map(|s| Move::from_ptn(s)).collect::<Vec<_>>();
        let mut tak = Tak::new();
        for move_ in move_sequence {
            tak.get_all_legal_moves(&mut legal_moves);
            assert!(legal_moves.contains(&Move::from_ptn(move_)));
            tak.make_move(Move::from_ptn(move_))
        }
        tak.get_all_legal_moves(&mut legal_moves);
        let predicted_legal_moves = legal_moves;
        for move_ in &legal_moves_known {
            assert!(predicted_legal_moves.contains(move_));
        }
        for move_ in &predicted_legal_moves {
            assert!(legal_moves_known.contains(move_))
        }
    }

    #[test]
    fn test_move_filtering() {
        let mut tak = Tak::new();
        let move_sequence = [ // looking forward 0 moves ahead, correctly identifies winning moves
            "f6", "a1",
            "e6", "f5",
            "d6", "f4",
            "c6", "f3",
            "c5", "f2",
            "c4",
        ].iter().map(|s| Move::from_ptn(s)).collect::<Vec<_>>();

        let expected_moves = [
            "f1", "Cf1"
        ].iter().map(|s| Move::from_ptn(s)).collect::<Vec<_>>();

        for move_ in move_sequence {
            tak.make_move(move_);
        }
        let (filtered_moves, score) = tak.get_filtered_moves(0, false);
        assert_eq!(score, 1);
        for move_ in &filtered_moves {
            assert!(expected_moves.contains(move_));
        }
        for move_ in &expected_moves {
            assert!(filtered_moves.contains(move_));
        }


        let mut tak = Tak::new(); // test looking 1 move ahead, correctly blocks road
        let move_sequence = [
            "f6", "a1",
            "e6", "f5",
            "d6", "f4",
            "c6", "f3",
            "c5", "f2",
            "c4", "e2",
        ].iter().map(|s| Move::from_ptn(s)).collect::<Vec<_>>();
        let expected_moves = [
            "e6>"
        ].iter().map(|s| Move::from_ptn(s)).collect::<Vec<_>>();
        for move_ in move_sequence {
            tak.make_move(move_);
        }
        let (filtered_moves, score) = tak.get_filtered_moves(1, false);
        assert_eq!(score, 0);
        for move_ in &filtered_moves {
            assert!(expected_moves.contains(move_));
        }
        for move_ in &expected_moves {
            assert!(filtered_moves.contains(move_));
        }


        let mut tak = Tak::new(); // test looking 2 moves ahead, setting up tenuie
        let move_sequence = [
            "a1", "f6",
            "f5", "b1",
            "f4", "a2",
            "f3", "b2",
            "e2", "a3",
        ].iter().map(|s| Move::from_ptn(s)).collect::<Vec<_>>();
        let expected_moves = [
            "f2", "Cf2", "e1", "Ce1"
        ].iter().map(|s| Move::from_ptn(s)).collect::<Vec<_>>();

        for move_ in move_sequence {
            tak.make_move(move_);
        }
        let (filtered_moves, score) = tak.get_filtered_moves(2, false);
        assert_eq!(score, 1);
        for move_ in &filtered_moves {
            assert!(expected_moves.contains(move_));
        }
        for move_ in &expected_moves {
            assert!(filtered_moves.contains(move_));
        }

        let mut tak = Tak::new(); // test complex position 3 moves ahead

        let move_sequence = [
            "f6", "a1",
            "d3", "Cd4",
            "c3", "e3",
            "Cc4", "e2",
            "e1", "d1",
            "c5", "c6",
            "b6", "b5",
            "c1", "e4",
            "d5", "d6",
            "c1>", "Sd2",
            "b6>", "e6",
            "c1", "d6<",
            "c2", "d6",
            "b6", "b5+",
            "c5+", "2b6>",
        ].iter().map(|s| Move::from_ptn(s)).collect::<Vec<_>>();

        let expected_moves = ["1d5+1", "Sa6", "Sb6"].iter().map(|s| Move::from_ptn(s)).collect::<Vec<_>>();

        for move_ in move_sequence {
            tak.make_move(move_);
        }
        let (filtered_moves, score) = tak.get_filtered_moves(3, false);
        assert_eq!(score, 0);
        for move_ in &filtered_moves {
            assert!(expected_moves.contains(move_));
        }
        for move_ in &expected_moves {
            assert!(filtered_moves.contains(move_));
        }
    }
}