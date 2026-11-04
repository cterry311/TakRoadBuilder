from engine_clients.roadbuilder import RoadBuilder
import tak_engine
import torch
import numpy as np
from model.tak_net import TakNet

import sqlite3

conn = sqlite3.connect("self_play_records/self_play_1.db")
conn.execute("PRAGMA journal_mode=WAL")
conn.executescript("""
CREATE TABLE IF NOT EXISTS self_play_games (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    seed INTEGER NOT NULL,            -- masked to < 2^63
    model TEXT NOT NULL,
    encoder_version INTEGER NOT NULL,
    komi INTEGER NOT NULL,
    max_sims INTEGER NOT NULL,
    batch_size INTEGER NOT NULL,
    c_puct REAL NOT NULL,
    urgency REAL NOT NULL,
    alpha REAL NOT NULL,
    eps REAL NOT NULL,
    temp_plies INTEGER NOT NULL,      -- how many of the first n plys get a randomly sampled move
    n_plies INTEGER NOT NULL, -- how many moves were made in the game
    outcome INTEGER NOT NULL,         -- 0 ongoing 1 white, 2 black, 3 draw
    end_reason TEXT
);
CREATE TABLE IF NOT EXISTS self_play_positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id INTEGER NOT NULL REFERENCES self_play_games(id),
    ply INTEGER NOT NULL,
    tps TEXT NOT NULL,
    move_raw INTEGER NOT NULL,        -- the move actually played
    sampled INTEGER NOT NULL,         -- 1 if sampled with temperature, 0 if argmax
    q REAL NOT NULL,                  -- root q, mover's view
    visits BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pos_game ON self_play_positions(game_id);
""")
conn.commit()

ENCODER_VERSION = 2


VISIT_DTYPE = np.dtype([("move", "<u2"), ("n", "<u4")])   # 2 + 4 = 6 bytes per entry

def pack_visits(visit_counts):
    """visit_counts: list of (move_raw, visits). Keeps only visited moves."""
    arr = np.array([(m, n) for m, n in visit_counts if n > 0], dtype=VISIT_DTYPE)
    return arr.tobytes()

def unpack_visits(blob):
    arr = np.frombuffer(blob, dtype=VISIT_DTYPE)
    return [(int(m), int(n)) for m, n in arr]          # back to a list of tuples


def sample_move(visit_counts, seed: int, T=0.2):
    rng = np.random.default_rng(seed)
    moves = np.array([m for m, n in visit_counts])
    visits = np.array([n for m, n in visit_counts], dtype=np.float64)
    p = visits ** (1.0 / T)
    p /= p.sum()
    return int(rng.choice(moves, p=p))

class Player:
    def __init__(self, model, komi, max_sims, batch_size, c_puct, urgency, alpha, eps, temp_plies):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        info = torch.load(model, map_location=self.device)
        self.model = TakNet(**info["params"])
        self.model.load_state_dict(info["weights"])
        self.model.to(self.device).eval()  # eval() matters: BatchNorm must not use batch stats

        self.BATCH = batch_size
        self.buf = np.zeros((self.BATCH, 25, 6, 6), dtype=np.float32)
        self.flat = self.buf.reshape(-1)  # a view onto buf; Rust writes through it

        self.komi = komi
        self.max_sims = max_sims
        self.batch_size = batch_size
        self.c_puct = c_puct
        self.urgency = urgency
        self.alpha = alpha
        self.eps = eps
        self.temp_plies = temp_plies

    @torch.no_grad()
    def gen_move(self, tps: str, ply: int, seed: int) -> tuple[int, list[tuple[int, int]], float]:
        # generate the move, return the following (move_raw, visit_counts, q), each entry in visit count is as follows (move_raw, visits)
        s = tak_engine.Search(tps, urgency=self.urgency, c_puct=self.c_puct, komi=self.komi)
        if s.is_over():
            return None
        i = 0
        while True:
            if s.sims() >= self.max_sims:
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
                s.add_noise(self.alpha, self.eps, seed)
            i += 1
        visit_counts = [(x[0], x[1]) for x in s.root_stats()]
        if ply < self.temp_plies:
            return sample_move(visit_counts, seed), visit_counts, s.root_q()
        else:
            best = max(visit_counts, key=lambda x: x[1])[0]
            return best, visit_counts, s.root_q()


def play_self(model, komi, max_sims, batch_size, c_puct, urgency, alpha, eps, temp_plies, max_ply, num_games):
    print('starting...')
    player = Player(model, komi, max_sims, batch_size, c_puct, urgency, alpha, eps, temp_plies)
    print('model loaded...')
    rand = np.random.default_rng()
    for _ in range(num_games):
        game = tak_engine.Tak(komi=komi)
        seed = int(rand.integers(0, 1 << 63))
        game_records = []
        i = 0
        end_reason = ''
        while True:
            tps = game.to_tps()
            mv, visit_counts, q = player.gen_move(tps, i, seed + i)
            game_records.append((tps, i, mv, visit_counts, q))
            game.make_move_raw(mv)
            i += 1
            if game.result() != 0:
                end_reason = 'game_over'
                break
            if i >= max_ply:
                end_reason = 'max_ply'
                break
        game_id = conn.execute('''
            INSERT INTO self_play_games (
                seed,   
                model,
                encoder_version,
                komi,
                max_sims,
                batch_size,
                c_puct,
                urgency,
                alpha,
                eps,
                temp_plies,    
                n_plies, 
                outcome,   
                end_reason
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            seed,
            model,
            ENCODER_VERSION,
            komi,
            max_sims,
            batch_size,
            c_puct,
            urgency,
            alpha,
            eps,
            temp_plies,
            i,
            game.result(),
            end_reason
        )).lastrowid
        for record in game_records:
            # (tps, i, mv, visit_counts, q)
            conn.execute('''
                INSERT INTO self_play_positions (
                    game_id,
                    ply,
                    tps,
                    move_raw,      
                    sampled,    
                    q,        
                    visits
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                game_id,
                record[1],
                record[0],
                record[2],
                1 if record[1] < temp_plies else 0,
                record[4],
                pack_visits(record[3])
            ))
        conn.commit()
        print(f'Game {game_id} written to database.')



if __name__ == "__main__":
    play_self('model_checkpoints/version_9_1.pt', 2, 5_000, 256, 1.5, -0.2, 0.05, 0.25, 20, 300, 1000)