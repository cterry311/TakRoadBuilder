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