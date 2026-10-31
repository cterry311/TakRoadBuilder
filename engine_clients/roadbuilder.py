
import numpy as np
import torch
import tak_engine
from model.tak_net import TakNet
import time
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
    def best_move(self, tps: str, max_sims=None, time_limit=None, c_puct=1.5, urgency=-0.2, alpha=0.1, eps=0.25, seed=None):
        if seed is None:
            seed = int(time.time() * 1000)
        if time_limit is not None:
            time_limit = time_limit / 1000.0
        s = tak_engine.Search(tps, urgency, c_puct)
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