"""
Engine client for Taktician
"""

import subprocess

class TakticianTEI:
    """
    A client for interacting with the Taktician engine.
    """
    def __init__(self, size: int=6, halfkomi: int=0, cmd=("taktician", "tei")):
        """
        Create a new Taktician client.
        :param size: the size of the board
        :param halfkomi: the komi of the game, half of the actual komi
        :param cmd: the path to the taktician executable and the initial command to send
        """
        self.size = size
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        self._send("tei")
        self._wait_for("teiok")
        self._send("isready")
        self._wait_for("readyok")
        self.new_game()

    def _send(self, line: str):
        """
        send a line to the taktition tei
        :param line: the line to send
        :return:
        """
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def _wait_for(self, prefix: str) -> str:
        """
        waits to recive a line starting with the given prefix
        :param prefix: the start of the line to wait for
        :return: the matching line
        """
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("engine exited")
            line = line.strip()
            if line.startswith(prefix):
                return line

    def new_game(self):
        """
        create a new game for taktition to play
        :return:
        """
        self._send(f"teinewgame {self.size}")
        self._send("isready")
        self._wait_for("readyok")

    def best_move(self, position: str=None, moves: list[str]=None, movetime_ms: int=1000) -> str:
        """
        gets taktitions best move with the given position, with the provided thinktime

        :param position: the position as a tps string
        :param moves: a list of moves as ptn strings
        :param movetime_ms: the time in milliseconds provided to taktition
        :return: the ptn move
        """
        if position is None:
            position = "startpos"
        else:
            position = f"tps {position}"
        request = "position "
        request += position
        if moves:
            request += " moves " + " ".join(moves)
        self._send(request)
        self._send(f"go movetime {movetime_ms}")
        return self._wait_for("bestmove").split()[1]

    def close(self):
        """
        close the connection with taktition
        :return:
        """
        self._send("quit")
        self.proc.wait(timeout=5)



if __name__ == "__main__":
    """
    test the Taktition engine
    """
    bot = TakticianTEI()
    bot.new_game()
    moves = []
    movetime_ms = 500
    while True:
        try:
            move = bot.best_move(moves=moves, movetime_ms=movetime_ms)
        except Exception:
            break
        moves.append(move)
        print(move)