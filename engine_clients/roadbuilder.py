
import numpy as np
import torch
import tak_engine
from model.tak_net import TakNet
import time
import sys
import threading
import queue
import socket
import websockets
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

info = torch.load("model_checkpoints/version_6_1.pt", map_location=device)
model = TakNet(**info["params"])
model.load_state_dict(info["weights"])
model.to(device).eval()                      # eval() matters: BatchNorm must not use batch stats

BATCH = 256
buf = np.zeros((BATCH, 25, 6, 6), dtype=np.float32)
flat = buf.reshape(-1)                       # a view onto buf; Rust writes through it

# warm up CUDA so the first real move doesn't pay kernel setup time
with torch.no_grad():
    model(torch.zeros(BATCH, 25, 6, 6, device=device))

@torch.no_grad()
def run_search(tps, max_sims=None, time_limit=None, c_puct=1.5, urgency=-0.2):
    s = tak_engine.Search(tps, urgency, c_puct)
    if s.is_over():
        return None

    start = time.perf_counter()
    while True:
        if max_sims is not None and s.sims() >= max_sims:
            break
        if time_limit is not None and time.perf_counter() - start >= time_limit:
            break

        n = s.collect(flat, BATCH)
        if n == 0:                           # every descent hit a finished game; already backed up
            continue

        x = torch.from_numpy(buf[:n]).to(device)
        policy, value = model(x)

        pol = torch.softmax(policy.flatten(1), dim=1).cpu().numpy().reshape(-1)
        vp = torch.softmax(value, dim=1)
        val = (vp[:, 2] - vp[:, 0]).cpu().numpy()

        s.submit(pol, val)                   # row i here must be row i of buf

    return s.best_move_raw(), s.root_stats(), s.sims()

tps = "1,x5/x,12,1211112C,x,2,12/12,x2,21C,2,112/x,1,1,1,x,2/x3,1,1,2/x3,1,x,2 1 24"
game = tak_engine.Tak.from_tps(tps)

start_time = time.perf_counter()
print('started search')
out = run_search(tps, time_limit=20.0)
print('finished search')
if out is not None:
    raw, stats, sims = out
    game.make_move_raw(raw)
    print(tak_engine.Tak.raw_to_ptn(raw), "after", sims, "sims")

print(time.perf_counter() - start_time)
'''

class RoadBuilder:
    def __init__(self, model_path, batch_size=256):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        info = torch.load(model_path, map_location=self.device)
        self.model = TakNet(**info["params"])
        self.model.load_state_dict(info["weights"])
        self.model.to(self.device).eval()  # eval() matters: BatchNorm must not use batch stats

        self.BATCH = batch_size
        self.buf = np.zeros((self.BATCH, 25, 6, 6), dtype=np.float32)
        self.flat = self.buf.reshape(-1)  # a view onto buf; Rust writes through it

        # warm up CUDA so the first real move doesn't pay kernel setup time
        with torch.no_grad():
            self.model(torch.zeros(self.BATCH, 25, 6, 6, device=self.device))

    @torch.no_grad()
    def best_move(self, tps: str, max_sims=None, time_limit=None, c_puct=1.5, urgency=-0.2, alpha=0.1, eps=0.25, seed=None, komi=2):
        if seed is None:
            seed = int(time.time() * 1000)
        if time_limit is not None:
            time_limit = time_limit / 1000.0
        s = tak_engine.Search(tps, urgency, c_puct, komi=komi)
        if s.is_over():
            return None

        start = time.perf_counter()
        # if max_sims is None:
        #     ply = int(tps.split(' ')[2])
        #     if ply <= 8:
        #         return self.best_move(tps, max_sims=2_000, time_limit=int(time_limit * 1000), c_puct=c_puct, urgency=urgency)
        i = 0
        while True:
            if max_sims is not None and s.sims() >= max_sims:
                break
            if time_limit is not None and time.perf_counter() - start >= time_limit:
                break

            n = s.collect(self.flat, self.BATCH)
            if n == 0:  # every descent hit a finished game; already backed up
                continue

            x = torch.from_numpy(self.buf[:n]).to(self.device)
            policy, value = self.model(x)

            pol = torch.softmax(policy.flatten(1), dim=1).cpu().numpy().reshape(-1)
            vp = torch.softmax(value, dim=1)
            val = (vp[:, 2] - vp[:, 0]).cpu().numpy()

            s.submit(pol, val)  # row i here must be row i of buf
            if i == 0:
                s.add_noise(alpha, eps, seed)
            i += 1
        return tak_engine.Tak.raw_to_ptn(s.best_move_raw()), s.root_stats(), s.sims(), s.root_q()


class RoadBuilderGame:
    def __init__(
            self,
            model_path,
            batch_size=256,
            c_puct=1.5,
            urgency=-0.2,
            alpha=0.1,
            eps=0.25,
            eval_cache=True,
            cache_max_age=2,
            komi=2
        ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        info = torch.load(model_path, map_location=self.device)
        self.model = TakNet(**info["params"])
        self.model.load_state_dict(info["weights"])
        self.model.to(self.device).eval()  # eval() matters: BatchNorm must not use batch stats

        self.BATCH = batch_size
        self.buf = np.zeros((self.BATCH, 25, 6, 6), dtype=np.float32)
        self.flat = self.buf.reshape(-1)  # a view onto buf; Rust writes through it

        # warm up CUDA so the first real move doesn't pay kernel setup time
        with torch.no_grad():
            self.model(torch.zeros(self.BATCH, 25, 6, 6, device=self.device))

        self.c_puct = c_puct
        self.urgency = urgency
        self.komi = komi

        self.game = tak_engine.Tak(komi=komi)
        self.tree = tak_engine.Search(self.game.to_tps(), urgency=urgency, c_puct=c_puct, komi=komi, eval_cache=eval_cache, cache_max_age=cache_max_age)
        self.root_noised = False

        self.alpha = alpha
        self.eps = eps

    def make_move(self, move):
        self.game.make_move(move)
        try:
            self.tree.rebase_tree(tak_engine.Tak.ptn_to_raw(move))
        except ValueError as e:
            self.tree = tak_engine.Search(self.game.to_tps(), urgency=self.urgency, c_puct=self.c_puct, komi=self.komi, eval_cache=True, cache_max_age=2)
    @torch.no_grad()
    def best_move(self, max_sims: int | None =None, time_limit: int | None =None):
        seed = int(time.time() * 1000)
        if max_sims is None and time_limit is None:
            raise ValueError("Either max_sims or time_limit must be specified")
        if time_limit is not None:
            time_limit = time_limit / 1000.0
        if max_sims is not None:
            max_sims = max_sims // 2
        start = time.perf_counter()
        i = 0
        while True:
            if max_sims is not None and self.tree.sims() >= max_sims:
                break
            if time_limit is not None and time.perf_counter() - start >= time_limit:
                break

            n = self.tree.collect(self.flat, self.BATCH)
            if n == 0:  # every descent hit a finished game; already backed up
                continue

            x = torch.from_numpy(self.buf[:n]).to(self.device)
            policy, value = self.model(x)

            pol = torch.softmax(policy.flatten(1), dim=1).cpu().numpy().reshape(-1)
            vp = torch.softmax(value, dim=1)
            val = (vp[:, 2] - vp[:, 0]).cpu().numpy()

            self.tree.submit(pol, val)  # row i here must be row i of buf
            if i == 0:
                self.tree.add_noise(self.alpha, self.eps, seed)
            i += 1
        return tak_engine.Tak.raw_to_ptn(self.tree.best_move_raw()), self.tree.root_stats(), self.tree.sims(), self.tree.root_q()
    @torch.no_grad()
    def bake_tree(self, max_sims=None, time_limit=None):
        if max_sims is None and time_limit is None:
            raise ValueError("Either max_sims or time_limit must be specified")
        if time_limit is not None:
            time_limit = time_limit / 1000.0
        if max_sims is not None:
            max_sims = max_sims // 2
        start = time.perf_counter()
        while True:
            if max_sims is not None and self.tree.sims() >= max_sims:
                break
            if time_limit is not None and time.perf_counter() - start >= time_limit:
                break

            n = self.tree.collect(self.flat, self.BATCH)
            if n == 0:  # every descent hit a finished game; already backed up
                continue

            x = torch.from_numpy(self.buf[:n]).to(self.device)
            policy, value = self.model(x)

            pol = torch.softmax(policy.flatten(1), dim=1).cpu().numpy().reshape(-1)
            vp = torch.softmax(value, dim=1)
            val = (vp[:, 2] - vp[:, 0]).cpu().numpy()

            self.tree.submit(pol, val)  # row i here must be row i of buf




def run_tei(get_lines=None, send=None):

    if get_lines is None:
        def get_lines():
            for line in sys.stdin:
                yield line

    lines = queue.Queue()

    def reader():
        for line in get_lines():  # blocks until a line arrives
            lines.put(line.strip())
        lines.put(None)  # sentinel: stdin closed (GUI quit/died)

    threading.Thread(target=reader, daemon=True).start()
    if send is None:
        def send(line):
            print(line, flush=True)

    state = {}

    while True:
        line: str = lines.get()
        if line is None:
            break
        line = line.strip()
        if line == 'tei':
            if state != {}:
                raise ValueError("Already in tei state")
            send('id name RoadBuilder')
            send('id author Cason Terry')
            send('option name HalfKomi type combo default 4 var 4')
            send('option name Urgency type spin default -20 min -300 max 300')
            send('option name CPuct type spin default 150 min 0 max 300')
            send('option name Alpha type spin default 10 min 0 max 100')
            send('option name Epsilon type spin default 25 min 0 max 100')
            send('teiok')
            state['tei'] = True
            state['Urgency'] = -0.2
            state['CPuct'] = 1.5
            state['Alpha'] = 0.1
            state['Epsilon'] = 0.25
        elif 'tei' not in state or not state['tei']:
            pass # the CLI is not in a tei state, ignore all commands that are not tei
        elif line.startswith('setoption '):
            parts = line.split(' ')
            if parts[1] == 'name':
                if parts[2] == 'HalfKomi':
                    if parts[4] != '4':
                        raise ValueError("Only HalfKomi=4 is supported")
                elif parts[2] =='Urgency':
                    state['Urgency'] = int(parts[4]) / 100
                elif parts[2] =='CPuct':
                    state['CPuct'] = int(parts[4]) / 100
                elif parts[2] =='Alpha':
                    state['Alpha'] = int(parts[4]) / 100
                elif parts[2] == 'Epsilon':
                    state['Epsilon'] = int(parts[4]) / 100
                else:
                    raise ValueError(f"Option {parts[2]} is not supported")
        elif line == 'isready':
            # assert that the model is ready to go, run a test batch
            send('readyok')
        elif line.startswith('teinewgame'):
            size = int(line.split(' ')[1])
            if size != 6:
                raise ValueError("Only 6x6 is supported")
            state['ingame'] = True
            state['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            info = torch.load('model_checkpoints/version_13_60.pt', map_location=state['device'])
            state['model'] = TakNet(**info["params"])
            state['model'].load_state_dict(info["weights"])
            state['model'].to(state['device']).eval()  # eval() matters: BatchNorm must not use batch stats

            state['BATCH'] = 256
            state['buf'] = np.zeros((state['BATCH'], 25, 6, 6), dtype=np.float32)
            state['flat'] = state['buf'].reshape(-1)  # a view onto buf; Rust writes through it

            # warm up CUDA so the first real move doesn't pay kernel setup time
            with torch.no_grad():
                state['model'](torch.zeros(state['BATCH'], 25, 6, 6, device=state['device']))
        elif line.startswith('position '):
            if 'ingame' not in state or not state['ingame']:
                raise ValueError("Not in a game")
            parts = line.split(' ')
            moves_index = 2
            if parts[1] == 'startpos':
                state['position'] = tak_engine.Tak(komi=2)
            elif parts[1] == 'tps':
                state['position'] = tak_engine.Tak.from_tps(parts[2], komi=2)
                moves_index = 3
            else:
                raise ValueError("expected 'startpos' or 'tps'")
            if len(parts) <= moves_index:
                pass # no moves from the position are specified
            elif parts[moves_index] == 'moves':
                moves = parts[moves_index + 1:]
                for move in moves:
                    state['position'].make_move(move.replace('*', ''))
            else:
                raise ValueError(f"unexpected token '{parts[moves_index]}'")
        elif line.startswith('go'):
            if 'position' not in state or not state['position']:
                raise ValueError("No position specified")
            parts = line.split(' ')
            if parts[1] == 'movetime':
                with torch.no_grad():
                    seed = int(time.time() * 1000)

                    s = tak_engine.Search(state['position'].to_tps(), komi=2, c_puct=state['CPuct'], urgency=state['Urgency'])
                    time_limit = int(parts[2]) / 1000.0
                    start = time.perf_counter()
                    i = 0
                    while True:
                        if time_limit is not None and time.perf_counter() - start >= time_limit:
                            break

                        n = s.collect(state['flat'], state['BATCH'])
                        if n == 0:  # every descent hit a finished game; already backed up
                            continue

                        x = torch.from_numpy(state['buf'][:n]).to(state['device'])
                        policy, value = state['model'](x)

                        pol = torch.softmax(policy.flatten(1), dim=1).cpu().numpy().reshape(-1)
                        vp = torch.softmax(value, dim=1)
                        val = (vp[:, 2] - vp[:, 0]).cpu().numpy()

                        s.submit(pol, val)  # row i here must be row i of buf
                        if i == 0:
                            s.add_noise(state['Alpha'], state['Epsilon'], seed)
                        i += 1
                        send(f'info depth {int(round(s.avg_depth()))} seldepth {s.max_depth()} nodes {s.sims()} score cp {int(round(s.root_q() * 100))} time {int((time.perf_counter() - start) * 1000)} pv {tak_engine.Tak.raw_to_ptn(s.best_move_raw())}')
                    best_move = tak_engine.Tak.raw_to_ptn(s.best_move_raw())
                    send(f"bestmove {best_move}")
            elif parts[1] == 'nodes':
                with torch.no_grad():
                    seed = int(time.time() * 1000)

                    s = tak_engine.Search(state['position'].to_tps(), komi=2, c_puct=state['CPuct'], urgency=state['Urgency'])
                    sim_limit = int(parts[2])
                    start = time.perf_counter()
                    i = 0
                    while True:
                        if s.sims() >= sim_limit:
                            break

                        n = s.collect(state['flat'], state['BATCH'])
                        if n == 0:  # every descent hit a finished game; already backed up
                            continue

                        x = torch.from_numpy(state['buf'][:n]).to(state['device'])
                        policy, value = state['model'](x)

                        pol = torch.softmax(policy.flatten(1), dim=1).cpu().numpy().reshape(-1)
                        vp = torch.softmax(value, dim=1)
                        val = (vp[:, 2] - vp[:, 0]).cpu().numpy()

                        s.submit(pol, val)  # row i here must be row i of buf
                        if i == 0:
                            s.add_noise(state['Alpha'], state['Epsilon'], seed)
                        i += 1
                        send(
                            f'info depth {int(round(s.avg_depth()))} seldepth {s.max_depth()} nodes {s.sims()} score cp {int(round(s.root_q() * 100))} time {int((time.perf_counter() - start) * 1000)} pv {tak_engine.Tak.raw_to_ptn(s.best_move_raw())}')
                    best_move = tak_engine.Tak.raw_to_ptn(s.best_move_raw())
                    send(f"bestmove {best_move}")
            elif parts[1] == 'infinite':
                with torch.no_grad():
                    seed = int(time.time() * 1000)

                    s = tak_engine.Search(state['position'].to_tps(), komi=2, c_puct=state['CPuct'], urgency=state['Urgency'])
                    start = time.perf_counter()
                    i = 0
                    while True:
                        try:
                            line = lines.get_nowait()
                            if line == 'stop':
                                break
                            else:
                                raise ValueError(f"unexpected token '{line}'")
                        except queue.Empty:
                            pass
                        n = s.collect(state['flat'], state['BATCH'])
                        if n == 0:  # every descent hit a finished game; already backed up
                            continue

                        x = torch.from_numpy(state['buf'][:n]).to(state['device'])
                        policy, value = state['model'](x)

                        pol = torch.softmax(policy.flatten(1), dim=1).cpu().numpy().reshape(-1)
                        vp = torch.softmax(value, dim=1)
                        val = (vp[:, 2] - vp[:, 0]).cpu().numpy()

                        s.submit(pol, val)  # row i here must be row i of buf
                        if i == 0:
                            s.add_noise(state['Alpha'], state['Epsilon'], seed)
                        i += 1
                        send(
                            f'info depth {int(round(s.avg_depth()))} seldepth {s.max_depth()} nodes {s.sims()} score cp {int(round(s.root_q() * 100))} time {int((time.perf_counter() - start) * 1000)} pv {tak_engine.Tak.raw_to_ptn(s.best_move_raw())}')
                    best_move = tak_engine.Tak.raw_to_ptn(s.best_move_raw())
                    send(f"bestmove {best_move}")
            elif parts[1] == 'wtime':
                raise NotImplementedError("dynamic movetime is not implemented")
            elif parts[1] == 'depth':
                with torch.no_grad():
                    seed = int(time.time() * 1000)

                    s = tak_engine.Search(state['position'].to_tps(), komi=2, c_puct=state['CPuct'], urgency=state['Urgency'])
                    depth_limit = int(parts[2])
                    start = time.perf_counter()
                    i = 0
                    while True:
                        if s.avg_depth() >= depth_limit:
                            break

                        n = s.collect(state['flat'], state['BATCH'])
                        if n == 0:  # every descent hit a finished game; already backed up
                            continue

                        x = torch.from_numpy(state['buf'][:n]).to(state['device'])
                        policy, value = state['model'](x)

                        pol = torch.softmax(policy.flatten(1), dim=1).cpu().numpy().reshape(-1)
                        vp = torch.softmax(value, dim=1)
                        val = (vp[:, 2] - vp[:, 0]).cpu().numpy()

                        s.submit(pol, val)  # row i here must be row i of buf
                        if i == 0:
                            s.add_noise(state['Alpha'], state['Epsilon'], seed)
                        i += 1
                        send(
                            f'info depth {int(round(s.avg_depth()))} seldepth {s.max_depth()} nodes {s.sims()} score cp {int(round(s.root_q() * 100))} time {int((time.perf_counter() - start) * 1000)} pv {tak_engine.Tak.raw_to_ptn(s.best_move_raw())}')
                    best_move = tak_engine.Tak.raw_to_ptn(s.best_move_raw())
                    send(f"bestmove {best_move}")
            else:
                raise ValueError(f"unknown token '{parts[1]}'")
        else:
            raise ValueError(f"unknown command '{line}'")




if __name__ == "__main__":

    # HOST, PORT = "localhost", 7731
    # sock = socket.create_connection((HOST, PORT))
    # f_write = sock.makefile("w", encoding="utf-8", newline="\n")
    # f_read = sock.makefile("r", encoding="utf-8", newline="\n")
    #
    # def send(line):
    #     print(">", line)
    #     f_write.write(line + "\n")
    #     f_write.flush()
    #
    # def get_lines():
    #     while True:
    #         line = f_read.readline()
    #         print("<", line)
    #         if not line:
    #             break
    #         yield line.strip()

    from websockets.sync.server import serve
    from websockets.exceptions import ConnectionClosed

    HOST, PORT = "localhost", 7731

    ws = None
    connected = threading.Event()
    finished = threading.Event()


    def handler(conn):
        global ws
        ws = conn
        connected.set()
        finished.wait()  # keep the connection open until we're done with it


    server = serve(handler, HOST, PORT)  # the port is open as soon as this returns
    threading.Thread(target=server.serve_forever, daemon=True).start()

    print(f"Listening on ws://{HOST}:{PORT}, waiting for a client...")
    connected.wait()
    print("Client connected")


    def send(line):
        print(">", line)
        ws.send(line)


    def get_lines():
        try:
            for message in ws:
                if isinstance(message, bytes):
                    message = message.decode("utf-8")
                print("<", message)
                yield message.strip()
        except ConnectionClosed:
            pass


    run_tei(get_lines=get_lines, send=send)
