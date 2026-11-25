"""
a client for interacting with the tiltak engine
"""
import subprocess
import os


class TeiEngine:
    def __init__(self, size: int=6, half_komi: int=0, exe: str=None):
        """
        Create a client for interacting with an arbitrary tei executable

        :param size: the size of the board
        :param half_komi: the half komi of the game
        :param exe: a string containing the name of the executable
        """
        here = os.path.dirname(os.path.abspath(__file__))
        exe = os.path.join(here, "exe", exe)
        self.p = subprocess.Popen([exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.send("tei"); self.wait_for("teiok")
        self.send(f"setoption name HalfKomi value {half_komi}")
        self.send(f"teinewgame {size}")
        self.send("isready"); self.wait_for("readyok")

    def send(self, line: str):
        """
        send a line to the tei engine

        :param line: the line to send
        :return:
        """
        self.p.stdin.write(line + "\n"); self.p.stdin.flush()

    def read_line(self) -> str:
        """
        reads the next line from the engine

        :return: the line as a string
        """
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError("engine exited")
        return line.strip()

    def wait_for(self, prefix: str) -> str:
        """
        wait for a line starting with the given prefix

        :param prefix: the prefix to wait for
        :return: the full line received
        """
        while True:
            line = self.read_line()
            if line.startswith(prefix):
                return line

    def send_position(self, position: str=None, moves: list[str]=None):
        """
        send a position to the enigne, prime it for getting the best move from that position

        :param position: the tps string of the position
        :param moves: a list of moves in ptn format to be applied from the position
        :return:
        """
        request = "position "
        if position is None:
            request += "startpos"
        else:
            request += f"tps {position}"
        if moves:
            request += f" moves {" ".join(moves)}"
        self.send(request)

    def best_move(self, position: str=None, moves: list[str]=None, movetime_ms: int=None, nodes: int=None, depth: int=None) -> str:
        """
        gets the best move from the engine for the given position

        :param position: the position as a tps string
        :param moves: a list of moves as ptn strings to be applied to the position
        :param movetime_ms: the time in milliseconds to search for
        :param nodes: the max number of nodes allowed to be searched
        :param depth: the max depth to search
        :return: the move in ptn format
        """
        self.send_position(position, moves)
        if movetime_ms is not None:
            self.send(f"go movetime {movetime_ms}")
        elif nodes is not None:
            self.send(f"go nodes {nodes}")
        elif depth is not None:
            self.send(f"go depth {depth}")
        else:
            raise ValueError('missing nodes and movetime argument, one is needed')
        return self.wait_for("bestmove").split()[1]

    def evaluate(self, position: str=None, moves: list[str]=None, movetime_ms: int=None, nodes: int=None, depth: int=None) -> dict:
        """Search a position and return Tiltak's evaluation of it.

        Everything is from the perspective of the SIDE TO MOVE. Don't pass a
        position where the game is already over: Tiltak panics and exits.

        :param position: the position as a tps string
        :param moves: a list of moves as ptn strings to be applied to the position
        :param movetime_ms: the time in milliseconds to search for
        :param nodes: the max number of nodes allowed to be searched
        :param depth: the max depth to search

        Returns a dict with:
            value:     (win - loss) / 1000 from the wdl estimate, in [-1, 1]
            score_cp:  centipawn-style score (None if Tiltak reported a mate)
            mate:      moves to mate if found, negative if being mated, else None
            wdl:       (win, draw, loss) per mille, or None
            depth:     search depth reached
            pv:        principal variation as a list of PTN moves
            best_move: the move Tiltak would play
        """
        self.send_position(position, moves)
        if movetime_ms is not None:
            self.send(f"go movetime {movetime_ms}")
        elif nodes is not None:
            self.send(f"go nodes {nodes}")
        elif depth is not None:
            self.send(f"go depth {depth}")
        else:
            raise ValueError('missing nodes and movetime argument, one is needed')

        info = {"value": None, "score_cp": None, "mate": None, "wdl": None, "depth": None, "pv": []}
        while True:
            line = self.read_line()
            if line.startswith("bestmove"):
                info["best_move"] = line.split()[1]
                break
            if line.startswith("info") and " score " in line:
                info.update(self._parse_info(line))

        if info["wdl"] is not None:
            win, _, loss = info["wdl"]
            info["value"] = (win - loss) / 1000
        elif info["mate"] is not None:
            info["value"] = 1.0 if info["mate"] > 0 else -1.0
        return info

    @staticmethod
    def _parse_info(line: str):
        """
        Parse a line like 'info depth 13 ... score cp -62 wdl 187 0 813 ... pv Cc4 d3'.

        :param line: the line to parse
        """
        tokens = line.split()
        out = {"score_cp": None, "mate": None, "wdl": None}
        i = 1
        while i < len(tokens):
            key = tokens[i]
            if key == "depth":
                out["depth"] = int(tokens[i + 1]); i += 2
            elif key == "score":
                kind, val = tokens[i + 1], int(tokens[i + 2])
                out["score_cp" if kind == "cp" else "mate"] = val
                i += 3
            elif key == "wdl":
                out["wdl"] = tuple(int(t) for t in tokens[i + 1:i + 4]); i += 4
            elif key == "pv":
                out["pv"] = tokens[i + 1:]
                break
            else:
                i += 2
        return out


    def close(self):
        """
        close the connection to the engine
        :return:
        """
        self.send("quit"); self.p.wait()