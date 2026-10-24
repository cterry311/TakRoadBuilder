"""Python client for the BeginnerBot Go engine (cmd/takengine).

Build the engine first, from the repo root:
    go build -o bin/takengine.exe ./cmd/takengine

Usage:
    from tak_engine import TakEngine

    with TakEngine("path/to/bin/takengine.exe") as eng:
        result = eng.best_move(moves=["a1", "e5"], size=5, time_ms=1000)
        print(result["move"], result["value"], result["pv"])
"""

import itertools
import json
import subprocess
import threading
import os
import sys


class TakEngineError(RuntimeError):
    pass


class BeginnerBotEngine:
    def __init__(self):
        here = os.path.dirname(os.path.abspath(__file__))
        exe = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "exe", "beginnerBot.exe")
        self._proc = subprocess.Popen(
            [exe],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._ids = itertools.count(1)
        self._lock = threading.Lock()

    def request(self, **fields):
        """Send a raw request dict; returns the parsed response dict."""
        fields["id"] = next(self._ids)
        with self._lock:
            self._proc.stdin.write(json.dumps(fields) + "\n")
            self._proc.stdin.flush()
            line = self._proc.stdout.readline()
        if not line:
            raise TakEngineError("engine process exited")
        resp = json.loads(line)
        if resp.get("error"):
            raise TakEngineError(resp["error"])
        return resp

    def best_move(self, moves=None, size=6, tps=None, time_ms=2000, depth=0,
                  seed=0, all_moves=False):
        """Ask the bot for its move.

        Give either `moves` (PTN moves from the start, e.g. ["a1", "e5", "c3"])
        with `size`, or a `tps` position string. `depth=0` means search as deep
        as possible within `time_ms`.
        """
        req = {"time_ms": time_ms, "depth": depth, "seed": seed,
               "all_moves": all_moves}
        if tps is not None:
            req["tps"] = tps
        else:
            req["size"] = size
            req["moves"] = list(moves or [])
        return self.request(**req)

    def close(self):
        if self._proc.poll() is None:
            self._proc.stdin.close()
            self._proc.wait(timeout=5)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


if __name__ == "__main__":
    import os
    import sys

    here = os.path.dirname(os.path.abspath(__file__))
    exe = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "exe", "beginnerBot.exe")

    # Self-play demo: the bot plays both sides.
    with BeginnerBotEngine() as eng:
        history = []
        while True:
            r = eng.best_move(moves=history, size=6, time_ms=300)
            if r["game_over"]:
                print("Game over, winner:", r["winner"] or "draw")
                break
            print(f"{len(history) + 1:3d}. {r['to_move']:5s} {r['move']:8s}"
                  f" value={r['value']} depth={r['depth']}")
            history.append(r["move"])
