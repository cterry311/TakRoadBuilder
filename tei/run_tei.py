import os
import glob
import time
import threading
import queue
import sys


def _system_path_entries():
    """PATH entries from this process plus the persisted Machine/User PATH in the
    registry, so CUDA added to PATH after the parent process started is still found."""
    entries = os.environ.get("PATH", "").split(os.pathsep)
    try:
        import winreg
        for root, key in ((winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
                          (winreg.HKEY_CURRENT_USER, r"Environment")):
            try:
                with winreg.OpenKey(root, key) as k:
                    val, _ = winreg.QueryValueEx(k, "Path")
                    entries += os.path.expandvars(val).split(os.pathsep)
            except OSError:
                pass
    except ImportError:
        pass
    return [e for e in entries if e]


def _find_dll_dir(dll_glob, extra_roots):
    """Return the first directory containing a file matching dll_glob, searching PATH
    entries (and their x64 / <cuda ver>\\x64 subfolders) and then extra_roots recursively."""
    for d in _system_path_entries():
        for cand in (d, os.path.join(d, "x64")):
            if glob.glob(os.path.join(cand, dll_glob)):
                return cand
        # cuDNN 9 layout: bin\<cuda version>\x64\cudnn64_9.dll -- prefer CUDA 13.x
        hits = sorted(glob.glob(os.path.join(d, "13.*", "x64", dll_glob)), reverse=True)
        if hits:
            return os.path.dirname(hits[0])
    for root in extra_roots:
        hits = sorted(glob.glob(os.path.join(root, "**", dll_glob), recursive=True), reverse=True)
        hits = [h for h in hits if "\\12." not in h] or hits  # prefer the CUDA 13 build
        if hits:
            return os.path.dirname(hits[0])
    return None


def setup_system_cuda():
    """Make the system-installed CUDA 13 + cuDNN 9 DLLs loadable by onnxruntime-gpu."""
    if os.name != "nt":
        return
    pf = os.environ.get("ProgramFiles", r"C:\Program Files")
    cuda_roots = [os.environ.get("CUDA_PATH", ""), os.path.join(pf, "NVIDIA GPU Computing Toolkit", "CUDA")]
    cudnn_roots = [os.environ.get("CUDNN_PATH", ""), os.path.join(pf, "NVIDIA", "CUDNN")]
    dirs = [
        _find_dll_dir("cublasLt64_13.dll", [r for r in cuda_roots if r]),
        _find_dll_dir("cudart64_13.dll", [r for r in cuda_roots if r]),
        _find_dll_dir("cudnn64_9.dll", [r for r in cudnn_roots if r]),
    ]
    for d in dict.fromkeys(d for d in dirs if d):
        os.add_dll_directory(d)
        os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
    if None in dirs:
        print("warning: could not locate CUDA 13 / cuDNN 9 DLLs; falling back to CPU", file=sys.stderr)


setup_system_cuda()

import tak_engine
import onnxruntime as ort
import numpy as np
from scipy.special import softmax


MODEL_PATH = r"C:\Users\cterry\PersonalCode\Capstone\tak\model_releases\release_1_1.onnx"



def get_lines():
    for line in sys.stdin:
        yield line

lines = queue.Queue()


def reader():
    for line in get_lines():  # blocks until a line arrives
        lines.put(line.strip())
    lines.put(None)  # sentinel: stdin closed (GUI quit/died)


threading.Thread(target=reader, daemon=True).start()

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
        pass  # the CLI is not in a tei state, ignore all commands that are not tei
    elif line.startswith('setoption '):
        parts = line.split(' ')
        if parts[1] == 'name':
            if parts[2] == 'HalfKomi':
                if parts[4] != '4':
                    raise ValueError("Only HalfKomi=4 is supported")
            elif parts[2] == 'Urgency':
                state['Urgency'] = int(parts[4]) / 100
            elif parts[2] == 'CPuct':
                state['CPuct'] = int(parts[4]) / 100
            elif parts[2] == 'Alpha':
                state['Alpha'] = int(parts[4]) / 100
            elif parts[2] == 'Epsilon':
                state['Epsilon'] = int(parts[4]) / 100
            else:
                raise ValueError(f"Option {parts[2]} is not supported")
    elif line == 'isready':
        # assert that the model is ready to go, run a test batch
        send('readyok')
    elif line.startswith('teinewgame'):
        sess = ort.InferenceSession(
            MODEL_PATH,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        state['model_sess'] = sess
        start_pos = tak_engine.Tak(komi=2)
        state['tree'] = tak_engine.Search(start_pos.to_tps(), urgency=state['Urgency'], c_puct=state['CPuct'], komi=2, capacity=200_000_000)
        state['position'] = start_pos.to_tps()
        state['moves'] = []
        state['ingame'] = True

        state['BATCH'] = 256
        state['buf'] = np.zeros((state['BATCH'], 25, 6, 6), dtype=np.float32)
        state['flat'] = state['buf'].reshape(-1)  # a view onto buf; Rust writes through it

        # warm up CUDA so the first real move doesn't pay kernel setup time
        sess.run(None, {"input": np.zeros((state['BATCH'], 25, 6, 6), dtype=np.float32)})
    elif line.startswith('position '):
        if 'ingame' not in state or not state['ingame']:
            raise ValueError("Not in a game")
        parts = line.split(' ')
        if parts[1] == 'startpos':
            tps_pos = tak_engine.Tak(komi=2).to_tps()
            move_index = 2
        elif parts[1] == 'tps':
            tps_pos = parts[2]
            move_index = 3
        else:
            raise ValueError("expected 'startpos' or 'tps'")
        reset_pos = tps_pos != state['position']
        state['position'] = tps_pos
        if reset_pos:
            if len(parts) >= (move_index + 2):
                temp_game = tak_engine.Tak.from_tps(tps_pos, komi=2)
                moves = parts[move_index + 1:]
                for move in moves:
                    temp_game.make_move(move.replace('*', ''))
                state['tree'] = tak_engine.Search(temp_game.to_tps(), urgency=state['Urgency'], c_puct=state['CPuct'], komi=2, capacity=200_000_000)
                state['moves'] = moves
                state['q'] = None
            else:
                state['tree'] = tak_engine.Search(tps_pos, urgency=state['Urgency'], c_puct=state['CPuct'], komi=2, capacity=200_000_000)
                state['moves'] = []
                state['q'] = None
        else:
            if len(parts) >= (move_index + 2):
                moves = parts[move_index + 1:]
                if len(state['moves']) > len(moves):
                    reset_pos = True
                for i in range(len(state['moves'])):
                    if reset_pos:
                        break
                    if state['moves'][i] != moves[i]:
                        reset_pos = True
                if reset_pos:
                    temp_game = tak_engine.Tak.from_tps(tps_pos, komi=2)
                    for move in moves:
                        temp_game.make_move(move.replace('*', ''))
                    state['tree'] = tak_engine.Search(temp_game.to_tps(), urgency=state['Urgency'], c_puct=state['CPuct'], komi=2, capacity=200_000_000)
                    state['moves'] = moves
                    state['q'] = None
                else:
                    for i in range(len(state['moves']), len(moves)):
                        try:
                            state['tree'].rebase_tree(tak_engine.Tak.ptn_to_raw(moves[i]))
                        except ValueError:
                            temp_game = tak_engine.Tak.from_tps(tps_pos, komi=2)
                            for move in moves:
                                temp_game.make_move(move.replace('*', ''))
                            state['tree'] = tak_engine.Search(temp_game.to_tps(), urgency=state['Urgency'], c_puct=state['CPuct'], komi=2, capacity=200_000_000)
                            break
                    state['moves'] = moves


    elif line.startswith('go'):
        if 'position' not in state or not state['position']:
            raise ValueError("No position specified")
        parts = line.split(' ')
        if parts[1] == 'movetime':
            seed = int(time.time() * 1000)

            time_limit = int(parts[2]) / 1000.0
            start = time.perf_counter()
            i = 0
            while True:
                if time_limit is not None and time.perf_counter() - start >= time_limit:
                    break

                n = state['tree'].collect(state['flat'], state['BATCH'])
                if n == 0:  # every descent hit a finished game; already backed up
                    continue

                x = state['buf'][:n]
                policy, value = state['model_sess'].run(None, {"input": x})

                pol = softmax(policy.reshape(policy.shape[0], -1), axis=1).reshape(-1)
                vp = softmax(value, axis=1)
                val = (vp[:, 2] - vp[:, 0])

                state['tree'].submit(pol, val)  # row i here must be row i of buf
                if i == 0:
                    state['tree'].add_noise(state['Alpha'], state['Epsilon'], seed)
                i += 1
                send(
                    f'info depth {int(round(state['tree'].avg_depth()))} seldepth {state['tree'].max_depth()} nodes {state['tree'].sims()} score cp {int(round(state['tree'].root_q() * 100))} time {int((time.perf_counter() - start) * 1000)} pv {tak_engine.Tak.raw_to_ptn(state['tree'].best_move_raw())}')
            best_move = tak_engine.Tak.raw_to_ptn(state['tree'].best_move_raw())
            state['q'] = state['tree'].root_q()
            send(f"bestmove {best_move}")
        elif parts[1] == 'nodes':
            seed = int(time.time() * 1000)

            node_limit = int(parts[2])
            start = time.perf_counter()
            i = 0
            while True:
                if node_limit is not None and state['tree'].sims() >= node_limit:
                    break

                n = state['tree'].collect(state['flat'], state['BATCH'])
                if n == 0:  # every descent hit a finished game; already backed up
                    continue

                x = state['buf'][:n]
                policy, value = state['model_sess'].run(None, {"input": x})

                pol = softmax(policy.reshape(policy.shape[0], -1), axis=1).reshape(-1)
                vp = softmax(value, axis=1)
                val = (vp[:, 2] - vp[:, 0])

                state['tree'].submit(pol, val)  # row i here must be row i of buf
                if i == 0:
                    state['tree'].add_noise(state['Alpha'], state['Epsilon'], seed)
                i += 1
                send(
                    f'info depth {int(round(state['tree'].avg_depth()))} seldepth {state['tree'].max_depth()} nodes {state['tree'].sims()} score cp {int(round(state['tree'].root_q() * 100))} time {int((time.perf_counter() - start) * 1000)} pv {tak_engine.Tak.raw_to_ptn(state['tree'].best_move_raw())}')
            best_move = tak_engine.Tak.raw_to_ptn(state['tree'].best_move_raw())
            state['q'] = state['tree'].root_q()
            send(f"bestmove {best_move}")
        elif parts[1] == 'infinite':
            seed = int(time.time() * 1000)

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

                n = state['tree'].collect(state['flat'], state['BATCH'])
                if n == 0:  # every descent hit a finished game; already backed up
                    continue

                x = state['buf'][:n]
                policy, value = state['model_sess'].run(None, {"input": x})

                pol = softmax(policy.reshape(policy.shape[0], -1), axis=1).reshape(-1)
                vp = softmax(value, axis=1)
                val = (vp[:, 2] - vp[:, 0])

                state['tree'].submit(pol, val)  # row i here must be row i of buf
                if i == 0:
                    state['tree'].add_noise(state['Alpha'], state['Epsilon'], seed)
                i += 1
                send(
                    f'info depth {int(round(state['tree'].avg_depth()))} seldepth {state['tree'].max_depth()} nodes {state['tree'].sims()} score cp {int(round(state['tree'].root_q() * 100))} time {int((time.perf_counter() - start) * 1000)} pv {tak_engine.Tak.raw_to_ptn(state['tree'].best_move_raw())}')
            best_move = tak_engine.Tak.raw_to_ptn(state['tree'].best_move_raw())
            state['q'] = state['tree'].root_q()
            send(f"bestmove {best_move}")
        elif parts[1] == 'wtime':
            seed = int(time.time() * 1000)

            temp_game = tak_engine.Tak.from_tps(state['position'], komi=2)
            for move in state['moves']:
                temp_game.make_move(move.replace('*', ''))

            increment_index = 6 if temp_game.side_to_move() == 'white' else 8
            time_index = 2 if temp_game.side_to_move() == 'white' else 4

            time_increment = int(parts[increment_index]) / 1000.0
            current_time = int(parts[time_index]) / 1000.0

            if 'q' in state and state['q'] is not None:
                q_modifier = (-0.8 * state['q']) + 1
            else:
                q_modifier = 1

            time_modifier = (current_time / 120) + 0.5

            requested_time = min(time_increment * time_modifier * q_modifier, 40)
            if requested_time <= current_time - 4:
                time_to_move = requested_time
            else:
                time_to_move = 2

            node_limit = int(parts[2])
            start = time.perf_counter()
            i = 0
            while True:
                if time.perf_counter() - start >= time_to_move:
                    break

                n = state['tree'].collect(state['flat'], state['BATCH'])
                if n == 0:  # every descent hit a finished game; already backed up
                    continue

                x = state['buf'][:n]
                policy, value = state['model_sess'].run(None, {"input": x})

                pol = softmax(policy.reshape(policy.shape[0], -1), axis=1).reshape(-1)
                vp = softmax(value, axis=1)
                val = (vp[:, 2] - vp[:, 0])

                state['tree'].submit(pol, val)  # row i here must be row i of buf
                if i == 0:
                    state['tree'].add_noise(state['Alpha'], state['Epsilon'], seed)
                i += 1
                send(
                    f'info depth {int(round(state['tree'].avg_depth()))} seldepth {state['tree'].max_depth()} nodes {state['tree'].sims()} score cp {int(round(state['tree'].root_q() * 100))} time {int((time.perf_counter() - start) * 1000)} pv {tak_engine.Tak.raw_to_ptn(state['tree'].best_move_raw())}')
            best_move = tak_engine.Tak.raw_to_ptn(state['tree'].best_move_raw())
            state['q'] = state['tree'].root_q()
            send(f"bestmove {best_move}")
        elif parts[1] == 'depth':
            seed = int(time.time() * 1000)

            depth_limit = int(parts[2])
            start = time.perf_counter()
            i = 0
            while True:
                if state['tree'].avg_depth() >= depth_limit:
                    break

                n = state['tree'].collect(state['flat'], state['BATCH'])
                if n == 0:  # every descent hit a finished game; already backed up
                    continue

                x = state['buf'][:n]
                policy, value = state['model_sess'].run(None, {"input": x})

                pol = softmax(policy.reshape(policy.shape[0], -1), axis=1).reshape(-1)
                vp = softmax(value, axis=1)
                val = (vp[:, 2] - vp[:, 0])

                state['tree'].submit(pol, val)  # row i here must be row i of buf
                if i == 0:
                    state['tree'].add_noise(state['Alpha'], state['Epsilon'], seed)
                i += 1
                send(
                    f'info depth {int(round(state['tree'].avg_depth()))} seldepth {state['tree'].max_depth()} nodes {state['tree'].sims()} score cp {int(round(state['tree'].root_q() * 100))} time {int((time.perf_counter() - start) * 1000)} pv {tak_engine.Tak.raw_to_ptn(state['tree'].best_move_raw())}')
            best_move = tak_engine.Tak.raw_to_ptn(state['tree'].best_move_raw())
            state['q'] = state['tree'].root_q()
            send(f"bestmove {best_move}")
        else:
            raise ValueError(f"unknown token '{parts[1]}'")
    elif line == 'quit':
        break
    else:
        raise ValueError(f"unknown command '{line}'")