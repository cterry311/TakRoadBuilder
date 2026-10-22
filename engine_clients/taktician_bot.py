import subprocess

class TakticianTEI:
    def __init__(self, size=6, cmd=("taktician", "tei")):
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

    def _send(self, line):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def _wait_for(self, prefix):
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("engine exited")
            line = line.strip()
            if line.startswith(prefix):
                return line

    def new_game(self):
        self._send(f"teinewgame {self.size}")
        self._send("isready")
        self._wait_for("readyok")

    def best_move(self, position=None, moves=None, movetime_ms=1000):
        """moves: list of PTN strings played so far, e.g. ['a1', 'a6', 'd3']"""
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
        self._send("quit")
        self.proc.wait(timeout=5)



if __name__ == "__main__":
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