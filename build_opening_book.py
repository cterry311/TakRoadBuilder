import numpy as np
import torch
import tak_engine
from model.tak_net import TakNet
import sqlite3

BATCH = 256
BUF = np.zeros((BATCH, 25, 6, 6), dtype=np.float32)
FLAT = BUF.reshape(-1)  # a view onto buf; Rust writes through it
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

info = torch.load('model_releases/version_13_60.pt', map_location=DEVICE)
MODEL = TakNet(**info["params"])
MODEL.load_state_dict(info["weights"])
MODEL.to(DEVICE).eval()  # eval() matters: BatchNorm must not use batch stats

@torch.no_grad()
def best_move(game: tak_engine.Tak, max_sims):
    tree = tak_engine.Search(game.to_tps(), komi=2)

    while True:
        if tree.sims() >= max_sims:
            break
        n = tree.collect(FLAT, BATCH)
        if n == 0:  # every descent hit a finished game; already backed up
            continue

        x = torch.from_numpy(BUF[:n]).to(DEVICE)
        policy, value = MODEL(x)

        pol = torch.softmax(policy.flatten(1), dim=1).cpu().numpy().reshape(-1)
        vp = torch.softmax(value, dim=1)
        val = (vp[:, 2] - vp[:, 0]).cpu().numpy()

        tree.submit(pol, val)  # row i here must be row i of buf


    stats = tree.root_stats()


    chosen_move = tak_engine.Tak.raw_to_ptn(tree.best_move_raw())

    return chosen_move, tree.root_stats(), tree.root_q(), tree.sims()


conn = sqlite3.connect('opening_book.db')
conn.execute('''
    CREATE TABLE IF NOT EXISTS opening_book (
        hash INT PRIMARY KEY,
        predicted_move TEXT,
        q_value FLOAT
    )
''')
conn.commit()

def save_move(game: tak_engine.Tak, depth, max_depth, max_sims, cuttoff=None):
    if cuttoff is None:
        cuttoff = [0.05] * (max_depth + 1)
    hash_exists = conn.execute('''
        SELECT COUNT(*) FROM opening_book WHERE hash = ?
    ''', (game.__hash__(), )).fetchone()[0]
    print(hash_exists, depth)
    if hash_exists:
        return
    move, stats, q, sims = best_move(game, max_sims)
    conn.execute('''
        INSERT INTO opening_book (hash, predicted_move, q_value)
        VALUES (?, ?, ?)
    ''', (game.__hash__(), move, q))
    conn.commit()
    if depth >= max_depth:
        return
    filtered_moves = [tak_engine.Tak.raw_to_ptn(m) for m, v, q in stats if v >= (sims * cuttoff[depth])]
    for f_move in filtered_moves:
        game.make_move(f_move)
        save_move(game, depth+1, max_depth, max_sims, cuttoff=cuttoff)
        game.unmake_move()

game = tak_engine.Tak(komi=2)
save_move(game, 0, 6, 1_000, cuttoff=[0.03, 0.03, 0.03, 0.04, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])