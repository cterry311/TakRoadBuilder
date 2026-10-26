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