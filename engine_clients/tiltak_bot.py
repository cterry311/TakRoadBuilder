import subprocess
import os
import sys

class Tiltak:
    def __init__(self, size=6, half_komi=0):
        here = os.path.dirname(os.path.abspath(__file__))
        exe = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "exe", "tiltak.exe")
        self.p = subprocess.Popen([exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.send("tei"); self.wait_for("teiok")
        self.send(f"setoption name HalfKomi value {half_komi}")
        self.send(f"teinewgame {size}")
        self.send("isready"); self.wait_for("readyok")

    def send(self, line):
        self.p.stdin.write(line + "\n"); self.p.stdin.flush()

    def wait_for(self, prefix):
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("engine exited")
            line = line.strip()
            if line.startswith(prefix):
                return line

    def best_move(self, position=None, moves=None, movetime_ms=1000):
        request = "position "
        if position is None:
            request += "startpos "
        else:
            request += f"tps {position}"
        if moves is not None:
            request += f"moves { " ".join(moves) }"
        self.send(request)
        self.send(f"go movetime {movetime_ms}")
        return self.wait_for("bestmove").split()[1]

    def close(self):
        self.send("quit"); self.p.wait()



if __name__ == "__main__":
    import os
    import sys


    bot = Tiltak(size=6, half_komi=0)

    moves = []
    while True:
        move = bot.best_move(moves=moves, movetime_ms=100)
        moves.append(move)
        print(move)