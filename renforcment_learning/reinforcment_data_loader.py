import sqlite3
import numpy as np
import torch
import tak_engine
from torch.utils.data import Dataset

# Must match the dtype used when writing. Defined here (not imported) because
# importing your self-play file would run its module-level DB setup.
VISIT_DTYPE = np.dtype([("move", "<u2"), ("n", "<u4")])
POLICY_LEN = 251 * 36


def split_game_ids(db_path, val_frac=0.05, seed=42):
    """Split by game, never by row. Capped games (outcome 0) are excluded."""
    conn = sqlite3.connect(db_path)
    ids = [r[0] for r in conn.execute("SELECT id FROM self_play_games WHERE outcome != 0")]
    conn.close()
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_frac))
    return ids[n_val:], ids[:n_val]          # train_ids, val_ids


class SelfPlayDataset(Dataset):
    def __init__(self, db_path, game_ids=None):
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        games = {g["id"]: dict(g) for g in
                 conn.execute("SELECT * FROM self_play_games WHERE outcome != 0")}
        if game_ids is not None:
            keep = set(game_ids)
            games = {k: v for k, v in games.items() if k in keep}
        if not games:
            raise ValueError("no usable games in database")
        self.games = games
        self.metadata = self._build_metadata()

        rows = conn.execute(
            "SELECT game_id, ply, tps, move_raw, sampled, q, visits FROM self_play_positions"
        ).fetchall()
        conn.close()
        self.rows = [tuple(r) for r in rows if r[0] in games]

    def _build_metadata(self):
        g = list(self.games.values())
        komis = {x["komi"] for x in g}
        versions = {x["encoder_version"] for x in g}
        if len(komis) != 1 or len(versions) != 1:
            raise ValueError(f"mixed settings: komi={komis}, encoder_version={versions}")
        return {
            "komi": komis.pop(),
            "encoder_version": versions.pop(),
            "models": sorted({x["model"] for x in g}),
            "max_sims": sorted({x["max_sims"] for x in g}),
            "n_games": len(g),
            "n_positions_expected": sum(x["n_plies"] for x in g),
        }

    @property
    def komi(self):
        return self.metadata["komi"]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        game_id, ply, tps, move_raw, sampled, q, blob = self.rows[i]
        outcome = self.games[game_id]["outcome"]       # 1 white, 2 black, 3 draw

        # policy target: normalized visits, dense over the 9036 layout
        v = np.frombuffer(blob, dtype=VISIT_DTYPE)
        idx = tak_engine.bulk_raw_to_index(v["move"].tolist())
        target = np.zeros(POLICY_LEN, dtype=np.float32)
        target[idx] = v["n"]
        target /= target.sum()

        # value class from the mover's view: 0 loss, 1 draw, 2 win
        mover = 1 if (ply & 1) == 0 else 2             # white moves on even plies
        if outcome == 3:
            value_class = 1
        else:
            value_class = 2 if outcome == mover else 0

        return (
            tps,                                       # str (collates to a list)
            torch.from_numpy(target),                  # (9036,) float32
            tak_engine.raw_to_index(move_raw),         # index of the move actually played
            value_class,                               # 0/1/2, matches the value head
            float(q),                                  # root q, mover's view
            int(sampled),
        )