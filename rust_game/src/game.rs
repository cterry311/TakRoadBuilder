// this file was made with significant assistance from AI

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

    /// remove the bottom `d` pieces from `self` and returns them as a `Stack`
    /// the `kind` of output will be `Kind::Flat` will reject the drop if the entire stack is dropped leaving height 0 behind
    #[inline] pub fn drop_bottom(&mut self, d: u8) -> Stack {
        debug_assert!(d > 0 && d < self.height);
        let r = self.height - d;                       // pieces that stay
        let dropped = Stack { bits: self.bits >> r, height: d, kind: Kind::Flat };
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
    pub fn from_ptn(ptn_move: String) -> Move {
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
            _ => 1,
        };
        if carry == 1 {
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
}

// ---------------------------------------------------------------- undo

#[derive(Clone, Copy, Debug)]
pub struct Undo {
    pub mv: Move,
    /// A capstone's final drop flattened a wall (restore Wall on undo).
    pub flattened: bool,
}

// ---------------------------------------------------------------- game

#[derive(Clone)]
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

        let converted_1 = Move::from_ptn(move_1.clone());
        let converted_2 = Move::from_ptn(move_2.clone());
        let converted_3 = Move::from_ptn(move_3.clone());

        let converted_4 = Move::from_ptn(move_4.clone());
        let converted_5 = Move::from_ptn(move_5.clone());
        let converted_6 = Move::from_ptn(move_6.clone());
        let converted_7 = Move::from_ptn(move_7.clone());

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
}