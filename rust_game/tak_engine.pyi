from numpy import ndarray

class Tak:
    """A Tak game on a 6x6 board.

    Positions are exchanged as TPS strings and moves as PTN strings.
    """

    def __init__(self) -> None:
        """Create a new game at the starting position."""

    @staticmethod
    def from_tps(tps: str) -> "Tak":
        """Build a game from a TPS string.

        Raises:
            ValueError: if the TPS string is malformed.
        """

    def to_tps(self) -> str:
        """Return the current position as a TPS string."""

    def make_move(self, ptn: str) -> None:
        """Play a move given in PTN, e.g. ``"a1"``, ``"Sc3"``, ``"3c4>12"``.

        Raises:
            ValueError: if the move string is invalid.
            RuntimeError: if the game is already over.
        """

    def unmake_move(self) -> None:
        """Undo the last move made through ``make_move``."""

    def legal_moves(self) -> list[str]:
        """Return every legal move as PTN strings."""

    def filtered_moves(self, depth: int, do_pruning: bool) -> tuple[list[str], int]:
        """Brute-force lookahead filter.

        Args:
            depth: extra plies searched beyond the immediate move (0 = only
                check for immediate wins and losses).
            do_pruning: stop at the first winning move if True.

        Returns:
            (best_moves, score) where score is 1 (win), 0 (draw/unknown)
            or -1 (loss) for the side to move.
        """

    def result(self) -> int:
        """0 ongoing, 1 white win, 2 black win, 3 draw."""

    def clone(self) -> "Tak":
        """Return an independent copy of the game."""

    def encode_into(self, out: ndarray):
        """encode the current board into the provided array, expected shape (25, 6, 6) or (900,)"""

    def to_tensor(self) -> ndarray:
        """returns a numpy array of shape (25, 6, 6) encoding game state"""

    def legal_moves_raw(self) -> list[int]:
        """gets all legal moves as 16 bit integers"""

    def filtered_moves_raw(self, depth: int, do_pruning: bool) -> tuple[list[int], int]:
        """gets filtered moves at given depth as 16 bit integers"""

    @staticmethod
    def ptn_to_raw(ptn: str) -> int:
        """accepts a move string in ptn format, returns a 16 bit integer"""

    @staticmethod
    def raw_to_ptn(mv: int) -> str:
        """accepts a move as a 16 bit integer and returns a ptn formated move string"""

    def get_legal_mask(self) -> np.ndarray:
        """gets a legal move mask"""


def encode_batch(games: List[Tak], buffer: np.ndarray):
    """
    Args:
        games: a list of Tak games
    buffer:
        a numpy array of shape (n_games, 25, 6, 6)
    accepts a list of games, mutates a buffer and replaces it with a set of tensor representations of the games
    """

def encode_training_batch(tps_strs: List[str], tensors: np.ndarray, masks: np.ndarray):
    """
    Args:
        tps_strs: a list of tps strings representations of game positions
        tensors: a numpy array to be mutated, it is of shape (n_games, 25, 6, 6)
        masks: a numpy array to be mutated, it is of shape (n_games, 251, 6, 6)
    buffer:
        a numpy array of shape (n_games, 25, 6, 6)
    accepts a list of games, mutates a buffer and replaces it with a set of tensor representations of the games
    """

def from_tps_bulk(tps_strs: List[str]) -> List[Tak]:
    """
    Args:
        tps_strs: a list of tps strings representations of game positions
    Returns:
        a list of Tak games
    """

def raw_to_index(raw: int) -> int:
    """
    Args:
       raw: a u16 encoding a move
    Returns:
        an integer indicating the index in the output space
    """

def bulk_raw_to_index(raws: List[int]) -> List[int]:
    """
    Args:
        raws: a list of u16 indicating moves
    Returns:
        a list of integers inidcating an index in the output space
    """

def index_to_raw(index: int) -> int:
    """take in an index, returns a u16 move encoding"""

def index_to_ptn(index: int) -> str:
    """take in an index, return a ptn move string"""


import numpy.typing as npt

class Search:
    """Batched PUCT Monte Carlo tree search from a single root position.

    The model runs in Python, and the tree lives in Rust. A search alternates
    between two calls:

        n = search.collect(buf, max_leaves)   # Rust picks leaves and fills buf
        ...run the model on buf[:n]...
        search.submit(policy, values)         # Rust expands the leaves and backs up

    Row i of the model output must correspond to row i of the buffer filled by
    the preceding collect call. Do not reorder, filter or shuffle rows.
    """

    def __init__(self, tps: str, urgency: float = -0.2, c_puct: float = 1.5) -> None:
        """
        Args:
            tps: TPS string of the position to search from (the root).
            urgency: First-play urgency offset. An unvisited child is scored as
                (parent's average value + urgency). Use 0 or a negative value;
                more negative focuses the search on moves that already look
                good, and a positive value pushes it toward trying every move.
            c_puct: Exploration constant. It scales the bonus
                c_puct * prior * sqrt(parent visits) / (1 + child visits), so
                larger values pull the search toward moves with high policy
                priors and few visits. Typical starting range is about 1 to 2.
        """

    def collect(self, buf: npt.NDArray[np.float32], max_leaves: int) -> int:
        """Descend the tree up to max_leaves times and encode each leaf reached.

        Leaves that are finished games are backed up immediately with their
        exact result and never reach the model. Duplicate selections of the
        same leaf within one batch are discarded, so the batch can be smaller
        than max_leaves.

        Args:
            buf: 1D float32 C-contiguous CPU array that Rust fills. Its length
                must be at least max_leaves * 900 (25 planes * 6 * 6). Row i
                occupies buf[i * 900:(i + 1) * 900]. Use a flat view of a
                (max_leaves, 25, 6, 6) array, e.g. arr.reshape(-1).
            max_leaves: Maximum number of leaves to collect.

        Returns:
            The number of rows written (n). It can be 0 if every descent ended
            on an already-finished game. In that case skip the model call and
            collect again. Only buf[:n] holds valid positions.

        Raises:
            ValueError: if buf is too small.
        """

    def submit(self, policy: npt.NDArray[np.float32], values: npt.NDArray[np.float32]) -> None:
        """Send model results for the leaves from the last collect call.

        Args:
            policy: 1D float32 array of length n * 9036, the softmaxed policy
                for each leaf (n rows of 251 * 36, flattened). Layout per row
                matches policy_index. Priors for each leaf's legal moves are
                renormalized in Rust, so softmaxing over all 9036 is fine.
            values: 1D float32 array of length n, each equal to
                p_win - p_loss from the value head, from the point of view of
                the side to move at that leaf (range -1 to 1).

        Raises:
            ValueError: if the lengths do not match the n returned by the last
                collect call.
        """

    def sims(self) -> int:
        """Number of simulations completed (the root's visit count).

        Only meaningful between submit and the next collect, because pending
        leaves temporarily inflate visit counts.
        """

    def is_over(self) -> bool:
        """True if the root position is already a finished game."""

    def best_move_raw(self) -> int:
        """Raw u16 encoding of the most-visited root move.

        Convert with Tak.raw_to_ptn. Check is_over() first, since a finished
        root has no moves.
        """

    def root_stats(self) -> list[tuple[int, int, float]]:
        """Statistics for every root move.

        Returns:
            A list of (raw_move, visits, value) tuples, where value is the
            average outcome from the root mover's point of view (-1 to 1). The
            visit counts are also the natural policy training target for
            self-play.
        """