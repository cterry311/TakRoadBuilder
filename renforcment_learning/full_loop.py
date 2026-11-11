# import nessesary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import numpy as np
import tak_engine
import sqlite3
import os
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import shutil
import random

# connect to google drive













# set up the classes and functions imported from other files



# TakNet


# CNN, has 6x6 input
# it has the following channels
# for the top layer, it has 6 channels, indicating flats, caps, and walls for each player, each on has a bianary representation, either 0 or 1
# there are 6 layers below that represented, each layer represented with 2 channels, one for white, and one for black.
# 1 layer encoding the stack height at that point, compressed within the range [0, 1)
# 1 layer for flat count differential
# 1 layer for each player representing if they have played their capstone yet
# 1 layer for each player representing the preportion of stones they have played.
# an all 1s plane for clearly identifying the board edge

# 251 policy output channels,
# 3 of those channels are place moves
# the rest are directional



IN_CHANNELS = 6 + (2 * 6) + 1 + 1 + 2 + 2 + 1

TRUNK_CHANNELS = 64
TRUNK_BLOCKS = 6

POLICY_OUTPUT_CHANNELS = 251
POLICY_BLOCKS = 1

VALUE_NET_SIZE = 64
VALUE_BLOCKS = 1



class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


class TakNet(nn.Module):
    def __init__(self, in_ch=IN_CHANNELS, tr_ch=TRUNK_CHANNELS, tr_bl=TRUNK_BLOCKS, pol_ch=POLICY_OUTPUT_CHANNELS, pol_bl=POLICY_BLOCKS, val_size=VALUE_NET_SIZE, val_bl=VALUE_BLOCKS,):
        super().__init__()
        trunk_ops: List[nn.Module] = [
            nn.Conv2d(in_ch, tr_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(tr_ch),
            nn.ReLU(),
        ]
        for _ in range(tr_bl):
            trunk_ops.append(ResBlock(tr_ch))
        self.trunk = nn.Sequential(*trunk_ops)

        policy_ops: List[nn.Module] = []
        for _ in range(pol_bl):
            policy_ops.append(ResBlock(tr_ch))
        policy_ops.append(nn.Conv2d(tr_ch, pol_ch, 1))
        self.policy = nn.Sequential(*policy_ops)

        value_ops: List[nn.Module] = [
            nn.Conv2d(tr_ch, 3, 1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * 6 * 6, val_size),
            nn.ReLU(),
        ]
        for _ in range(val_bl):
            value_ops.append(nn.Linear(val_size, val_size))
            value_ops.append(nn.ReLU())
        value_ops.append(nn.Linear(val_size, 3))
        # value_ops.append(nn.Softmax(dim=1)) #value head is now logits instead of probabilities
        self.value = nn.Sequential(*value_ops)

    def forward(self, x):
        x = self.trunk(x)
        return self.policy(x), self.value(x) # value output, 0 = loss, 1 = draw, 2 = win









# Must match the dtype used when writing. Defined here (not imported) because
# importing your self-play file would run its module-level DB setup.
VISIT_DTYPE = np.dtype([("move", "<u2"), ("n", "<u4")])
POLICY_LEN = 251 * 36
INPUT_LEN = 25 * 36


def split_game_ids(db_path, val_frac=0.05, seed=42):
    """Split by game, never by row. Capped games (outcome 0) are excluded."""
    conn = sqlite3.connect(db_path)
    ids = [r[0] for r in conn.execute("SELECT id FROM self_play_games WHERE outcome != 0")]
    conn.close()
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_frac))
    return ids[n_val:], ids[:n_val]          # train_ids, val_ids


def load_selfplay_tensors(db_path, game_ids, device, chunk=4096):
    """Encode every position once and keep the whole set on `device`.

    Replaces the old per-row Dataset: encoding happens once instead of every
    epoch, and batches are plain tensor indexing on the GPU, so there's no
    DataLoader / Python work per sample.

    The policy target is kept sparse (visited moves only) and made dense per
    batch with dense_targets(), which saves ~36 KB of memory per position.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    games = {g["id"]: dict(g) for g in
             conn.execute("SELECT * FROM self_play_games WHERE outcome != 0")}
    keep = set(game_ids)
    games = {k: v for k, v in games.items() if k in keep}
    if not games:
        raise ValueError("no usable games in database")

    g = list(games.values())
    komis = {x["komi"] for x in g}
    versions = {x["encoder_version"] for x in g}
    if len(komis) != 1 or len(versions) != 1:
        raise ValueError(f"mixed settings: komi={komis}, encoder_version={versions}")
    komi = komis.pop()
    metadata = {
        "komi": komi,
        "encoder_version": versions.pop(),
        "models": sorted({x["model"] for x in g}),
        "max_sims": sorted({x["max_sims"] for x in g}),
        "n_games": len(g),
    }

    rows = conn.execute(
        "SELECT game_id, ply, tps, move_raw, sampled, q, visits FROM self_play_positions"
    ).fetchall()
    conn.close()
    rows = [tuple(r) for r in rows if r[0] in games]
    n = len(rows)
    metadata["n_positions"] = n

    # inputs and legal masks, encoded in chunks by the Rust engine
    x = torch.empty((n, 25, 6, 6), dtype=torch.float32)
    x_np = x.numpy()
    legal = torch.empty((n, POLICY_LEN), dtype=torch.bool)
    mask_buf = np.empty((chunk, 251, 6, 6), dtype=np.uint8)
    for s in range(0, n, chunk):
        tps = [r[2] for r in rows[s:s + chunk]]
        m = len(tps)
        tak_engine.encode_training_batch(tps, x_np[s:s + m], mask_buf[:m], komi=komi)
        legal[s:s + m] = torch.from_numpy(mask_buf[:m].reshape(m, -1)).bool()

    # sparse policy targets: normalized visit counts, padded to the longest list.
    # padding is index 0 with weight 0, which scatter_add ignores
    visits = [np.frombuffer(r[6], dtype=VISIT_DTYPE) for r in rows]
    k_max = max(len(v) for v in visits)
    tgt_idx = np.zeros((n, k_max), dtype=np.int64)
    tgt_val = np.zeros((n, k_max), dtype=np.float32)
    for i, v in enumerate(visits):
        k = len(v)
        counts = v["n"].astype(np.float32)
        tgt_idx[i, :k] = tak_engine.bulk_raw_to_index(v["move"].tolist())
        tgt_val[i, :k] = counts / counts.sum()

    # value class from the mover's view: 0 loss, 1 draw, 2 win
    outcome = np.array([games[r[0]]["outcome"] for r in rows])      # 1 white, 2 black, 3 draw
    mover = np.where(np.array([r[1] for r in rows]) % 2 == 0, 1, 2)  # white moves on even plies
    vclass = np.where(outcome == 3, 1, np.where(outcome == mover, 2, 0))

    data = {
        "x": x,
        "legal": legal,
        "tgt_idx": torch.from_numpy(tgt_idx),
        "tgt_val": torch.from_numpy(tgt_val),
        "played": torch.tensor(tak_engine.bulk_raw_to_index([r[3] for r in rows]), dtype=torch.long),
        "vclass": torch.from_numpy(vclass).long(),
        "q": torch.tensor([r[5] for r in rows], dtype=torch.float32),       # root q, mover's view
        "sampled": torch.tensor([r[4] for r in rows], dtype=torch.float32),
    }
    return {k: v.to(device) for k, v in data.items()}, metadata


def dense_targets(data, idx):
    """(len(idx), 9036) policy targets for the rows in idx."""
    target = torch.zeros((len(idx), POLICY_LEN), device=data["tgt_val"].device)
    return target.scatter_add_(1, data["tgt_idx"][idx], data["tgt_val"][idx])
















# method for running self play

# ---------------- settings ----------------
SELF_PLAY_DB = "self_play_temp.db"
SP_KOMI = 2
MAX_SIMS = 1_000
N_PARALLEL = 64          # games played at the same time; their leaves share one GPU batch
LEAVES_PER_GAME = 32     # leaves collected from each tree per round (was 256 from a single tree)
C_PUCT = 1.5
URGENCY = -0.2
ALPHA = 0.05
EPS = 0.25
TEMP_PLIES = 14
MAX_PLY = 300
NUM_GAMES = 300
EVAL_CACHE = True        # reuse model evals for repeated / symmetric positions within a game
CACHE_MAX_AGE = 2        # drop cache entries unused for this many moves
TEMP = 0.5


def setup_db(db_path=SELF_PLAY_DB):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
    DROP INDEX IF EXISTS idx_pos_game;
    DROP TABLE IF EXISTS self_play_games;
    DROP TABLE IF EXISTS self_play_positions;

    CREATE TABLE IF NOT EXISTS self_play_games (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        seed INTEGER NOT NULL,            -- masked to < 2^63
        model TEXT NOT NULL,
        encoder_version INTEGER NOT NULL,
        komi INTEGER NOT NULL,
        max_sims INTEGER NOT NULL,
        batch_size INTEGER NOT NULL,      -- leaves collected per tree per round
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
    return conn

ENCODER_VERSION = 2


VISIT_DTYPE = np.dtype([("move", "<u2"), ("n", "<u4")])   # 2 + 4 = 6 bytes per entry

def pack_visits(visit_counts):
    """visit_counts: list of (move_raw, visits). Keeps only visited moves."""
    arr = np.array([(m, n) for m, n in visit_counts if n > 0], dtype=VISIT_DTYPE)
    return arr.tobytes()

def unpack_visits(blob):
    arr = np.frombuffer(blob, dtype=VISIT_DTYPE)
    return [(int(m), int(n)) for m, n in arr]          # back to a list of tuples


def sample_move(visit_counts, seed: int, T=TEMP):
    rng = np.random.default_rng(seed)
    moves = np.array([m for m, n in visit_counts])
    visits = np.array([n for m, n in visit_counts], dtype=np.float64)
    p = visits ** (1.0 / T)
    p /= p.sum()
    return int(rng.choice(moves, p=p))


def copy_db(conn: sqlite3.Connection, copy_path):
    other = sqlite3.connect(copy_path)
    conn.backup(other)
    other.close()


class Slot:
    """One game in progress, plus the search for its current move."""
    def __init__(self, komi, seed):
        self.game = tak_engine.Tak(komi=komi)
        self.seed = seed
        self.ply = 0
        self.records = []        # (tps, ply, move_raw, visit_counts, q)
        self.search = None       # one Search per game, rebased onto each move played
        self.noised = False      # Dirichlet noise added at the current root


class Group:
    """A set of games whose leaves go to the GPU together.

    Each group has its own pinned input/output buffers, so one group can be
    on the GPU while the CPU collects leaves for the other.
    """
    def __init__(self, max_rows, pin):
        self.slots = []
        self.spans = []          # (slot, row offset, n rows) from the last collect
        self.collected = False   # True between collect and finish
        self.event = None        # CUDA event marking this group's results as ready
        self.inputs = torch.zeros((max_rows, 25, 6, 6), dtype=torch.float32, pin_memory=pin)
        self.inputs_flat = self.inputs.numpy().reshape(-1)        # Rust writes through this view
        self.policy = torch.empty((max_rows, POLICY_LEN), dtype=torch.float32, pin_memory=pin)
        self.policy_flat = self.policy.numpy().reshape(-1)
        self.values = torch.empty(max_rows, dtype=torch.float32, pin_memory=pin)
        self.values_np = self.values.numpy()


class SelfPlayRunner:
    """Plays many games at once and batches every game's leaves into one model call.

    Speedups over playing one game at a time:
      * GPU batches are large (up to N_PARALLEL * LEAVES_PER_GAME / 2 rows) and
        full, instead of whatever was left of 256 after duplicate leaves were dropped.
      * Games are split into two groups. While the GPU evaluates one group, the
        CPU (Rust collect/submit) works on the other.
      * collect_batch / submit_batch search every tree in a group in parallel,
        one tree per CPU core, with the GIL released.
      * The forward pass runs in float16 autocast with channels_last, and transfers
        use pinned memory so they can overlap with other work.
    """
    def __init__(self, model_path, komi, max_sims, n_parallel, leaves_per_game,
                 c_puct, urgency, alpha, eps, temp_plies, max_ply):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cuda = self.device.type == "cuda"

        info = torch.load(model_path, map_location=self.device)
        self.model = TakNet(**info["params"])
        self.model.load_state_dict(info["weights"])
        self.model.to(self.device).eval()  # eval() matters: BatchNorm must not use batch stats
        # channels_last makes the convs about 1.5x faster on tensor-core GPUs
        self.model.to(memory_format=torch.channels_last)

        self.model_path = model_path
        self.komi = komi
        self.max_sims = max_sims
        self.n_parallel = n_parallel
        self.leaves = leaves_per_game
        self.c_puct = c_puct
        self.urgency = urgency
        self.alpha = alpha
        self.eps = eps
        self.temp_plies = temp_plies
        self.max_ply = max_ply

        per_group = (n_parallel + 1) // 2
        self.groups = [Group(per_group * leaves_per_game, self.cuda) for _ in range(2)]
        self.per_group = per_group

    # ---- per game ----
    def new_search(self, slot):
        slot.search = tak_engine.Search(slot.game.to_tps(), urgency=self.urgency,
                                        c_puct=self.c_puct, komi=self.komi,
                                        eval_cache=EVAL_CACHE, cache_max_age=CACHE_MAX_AGE)
        slot.noised = False

    def add_noise(self, slot):
        """Noise goes on the root's children, so the root must already be expanded."""
        slot.search.add_noise(self.alpha, self.eps, slot.seed + slot.ply)
        slot.noised = True

    def play_move(self, slot):
        """Search is done: pick and play a move. Returns an end reason, or None if the game goes on.

        When the game goes on, the search is rebased onto the move played, so the
        visits already spent below that move carry over to the next search.
        """
        s = slot.search
        visit_counts = [(x[0], x[1]) for x in s.root_stats()]
        if slot.ply < self.temp_plies:
            mv = sample_move(visit_counts, slot.seed + slot.ply)
        else:
            mv = max(visit_counts, key=lambda x: x[1])[0]
        slot.records.append((slot.game.to_tps(), slot.ply, mv, visit_counts, s.root_q()))
        slot.game.make_move_raw(mv)
        slot.ply += 1
        if slot.game.result() != 0:
            return 'game_over'
        if slot.ply >= self.max_ply:
            return 'max_ply'
        s.rebase_tree(mv)
        slot.noised = False
        if s.sims() > 0:          # new root was already expanded: noise can go on right away
            self.add_noise(slot)
        return None

    # ---- per group ----
    def collect(self, group):
        """CPU: every game in the group descends its tree and writes leaves into the group's buffer.

        The trees are searched in parallel in Rust; rows come back packed in slot order.
        """
        group.spans = []
        if group.slots:
            counts = tak_engine.collect_batch([slot.search for slot in group.slots],
                                              group.inputs_flat, self.leaves)
            off = 0
            for slot, n in zip(group.slots, counts):
                group.spans.append((slot, off, n))
                off += n
        group.collected = bool(group.spans)
        return sum(n for _, _, n in group.spans)

    @torch.no_grad()
    def launch(self, group, n):
        """GPU: queue the model call for n rows. Returns without waiting for the result."""
        x = group.inputs[:n].to(self.device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.cuda):
            policy, value = self.model(x)
        pol = torch.softmax(policy.float().flatten(1), dim=1)
        vp = torch.softmax(value.float(), dim=1)
        group.policy[:n].copy_(pol, non_blocking=True)
        group.values[:n].copy_(vp[:, 2] - vp[:, 0], non_blocking=True)
        if self.cuda:
            group.event = torch.cuda.Event()
            group.event.record()

    def finish(self, group, start_game, write_game):
        """Wait for the group's results, hand them to each search, and advance finished searches."""
        if not group.collected:
            return
        if group.event is not None:
            group.event.synchronize()
            group.event = None
        total = sum(n for _, _, n in group.spans)
        tak_engine.submit_batch([slot.search for slot, _, _ in group.spans],
                                group.policy_flat[:total * POLICY_LEN], group.values_np[:total])
        kept = []
        for slot, off, n in group.spans:
            s = slot.search
            if not slot.noised and s.sims() > 0:   # first submit at this root expanded it
                self.add_noise(slot)
            if s.sims() >= self.max_sims:
                end_reason = self.play_move(slot)
                if end_reason is not None:
                    write_game(slot, end_reason)
                    slot = start_game()
                    if slot is None:
                        continue
            kept.append(slot)
        group.slots = kept
        group.spans = []
        group.collected = False

    def run(self, num_games, conn, save_path):
        rand = np.random.default_rng()
        started = 0
        finished = 0

        def start_game():
            nonlocal started
            if started >= num_games:
                return None
            started += 1
            slot = Slot(self.komi, int(rand.integers(0, 1 << 63)))
            self.new_search(slot)
            return slot

        def write_game(slot, end_reason):
            nonlocal finished
            game_id = conn.execute('''
                INSERT INTO self_play_games (
                    seed, model, encoder_version, komi, max_sims, batch_size, c_puct,
                    urgency, alpha, eps, temp_plies, n_plies, outcome, end_reason
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                slot.seed, self.model_path, ENCODER_VERSION, self.komi, self.max_sims, self.leaves,
                self.c_puct, self.urgency, self.alpha, self.eps, self.temp_plies,
                slot.ply, slot.game.result(), end_reason
            )).lastrowid
            # record = (tps, ply, mv, visit_counts, q)
            conn.executemany('''
                INSERT INTO self_play_positions (game_id, ply, tps, move_raw, sampled, q, visits)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', [
                (game_id, ply, tps, mv, 1 if ply < self.temp_plies else 0, q, pack_visits(visits))
                for tps, ply, mv, visits, q in slot.records
            ])
            conn.commit()
            finished += 1
            print(f'Game {game_id} written to database ({slot.ply} plies, {end_reason}).')
            if finished % 10 == 0:
                copy_db(conn, f'self_play_records/{save_path}')

        for i in range(self.n_parallel):
            slot = start_game()
            if slot is None:
                break
            self.groups[i % 2].slots.append(slot)

        # pipeline: collect A, launch A, finish B, collect B, launch B, finish A, ...
        # the GPU works on one group while the CPU finishes and collects the other
        while any(g.slots or g.collected for g in self.groups):
            for g, other in ((0, 1), (1, 0)):
                group = self.groups[g]
                n = self.collect(group)
                if n:
                    self.launch(group, n)
                self.finish(self.groups[other], start_game, write_game)

        copy_db(conn, f'self_play_records/{save_path}')


def run_self_play(model_path, save_path):
    model_path = f"model_checkpoints/{model_path}"
    conn = setup_db()
    print('starting...')
    runner = SelfPlayRunner(model_path, SP_KOMI, MAX_SIMS, N_PARALLEL, LEAVES_PER_GAME,
                            C_PUCT, URGENCY, ALPHA, EPS, TEMP_PLIES, MAX_PLY)
    print('model loaded...')
    runner.run(NUM_GAMES, conn, save_path)
    print(f'finished self play with model {model_path}')
    conn.commit()
    conn.close()














# method for doing the renforcment learning


# ---------------- settings ----------------


EPOCHS = 3
BATCH = 128
LR = 3e-4                             # lower than supervised (guess, tune)
DECAY = 3e-2
W_V = 0.5                             # value cross-entropy on game outcome
GAMMA = 0.5                          # value vs search q
BETA = 0.0                            # advantage term; set 0 to disable
ADV_CLIP = 1.0
VAL_FRAC = 0.05
ENCODER_VERSION = 2                   # must match what the data was generated with

WINDOW_SIZE = 15                       # the size of the replay buffer, will use training examples from the past n generations of model
N_GAMES_SAMPLED = 1000                 # the number of games sampled from the window, or the size of the window if the window is not large enough to accomonate

# ---------------- augmentation (copied from supervised file) ----------------
DIRS = [(1, 0), (-1, 0), (0, -1), (0, 1)]          # engine order + - < >
TRANSFORMS = [(k, flip) for flip in (False, True) for k in range(4)]

def apply_T(t, k, flip):
    t = torch.rot90(t, k, dims=(-2, -1))
    return torch.flip(t, dims=(-1,)) if flip else t

def build_perms():
    base = torch.arange(36).reshape(6, 6)
    perms = []
    for k, flip in TRANSFORMS:
        moved = apply_T(base, k, flip).flatten()
        new_of_old = torch.empty(36, dtype=torch.long)
        new_of_old[moved] = torch.arange(36)
        a = divmod(int(new_of_old[2 * 6 + 2]), 6)
        dir_map = []
        for dy, dx in DIRS:
            b = divmod(int(new_of_old[(2 + dy) * 6 + 2 + dx]), 6)
            dir_map.append(DIRS.index((b[0] - a[0], b[1] - a[1])))
        perm = torch.empty(251 * 36, dtype=torch.long)
        for ch in range(251):
            if ch < 3:
                nch = ch
            else:
                s = ch - 3
                nch = 3 + (s // 4) * 4 + dir_map[s % 4]
            for cell in range(36):
                perm[ch * 36 + cell] = nch * 36 + int(new_of_old[cell])
        perms.append(perm)
    return perms

def augment(x, legal, target, played, perms):
    """x (n,25,6,6); legal (n,9036) bool; target (n,9036) float; played (n,) int64. All on one device.
    Value targets are invariant, so they are not touched."""
    n = x.shape[0]
    ts = torch.randint(0, 8, (n,), device=x.device)
    x_o, legal_o = x.clone(), torch.empty_like(legal)
    target_o, played_o = torch.empty_like(target), played.clone()
    for t, (k, flip) in enumerate(TRANSFORMS):
        idx = (ts == t).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        perm = perms[t]
        x_o[idx] = apply_T(x[idx], k, flip)
        legal_o[idx[:, None], perm[None, :]] = legal[idx]       # new[perm[i]] = old[i]
        target_o[idx[:, None], perm[None, :]] = target[idx]
        played_o[idx] = perm[played[idx]]
    return x_o, legal_o, target_o, played_o

# ---------------- loss ----------------
def compute_losses(policy, value_logits, legal, target, played, value_class, q, sampled):
    # losses run in float32 even under autocast: -1e9 overflows float16 and would give NaNs
    policy, value_logits = policy.float(), value_logits.float()
    logits = policy.flatten(1).masked_fill(~legal, -1e9)
    logp = F.log_softmax(logits, dim=1)

    policy_loss = -(target * logp).sum(1).mean()
    value_ce = F.cross_entropy(value_logits, value_class)
    probs = F.softmax(value_logits, dim=1)
    value_q = F.mse_loss(probs[:, 2] - probs[:, 0], q)

    # advantage term: z - q, from the mover's view, no gradient through the advantage
    z = value_class.float() - 1.0                              # 0/1/2 -> -1/0/+1
    A = (z - q).clamp(-ADV_CLIP, ADV_CLIP).detach()
    logp_a = logp.gather(1, played[:, None]).squeeze(1)
    p_a = logp_a.exp().clamp(max=1 - 1e-6)
    pos = A.clamp(min=0) * (-logp_a)                           # won more than expected: raise move
    neg = (-A).clamp(min=0) * (-torch.log1p(-p_a))             # lost more than expected: lower move


    #adv = ((pos + neg) * sampled).mean()                       # sampled moves only
    adv = (pos + neg).mean()  # all moves have this loss applied, not just sampled


    total = policy_loss + W_V * value_ce + GAMMA * value_q + BETA * adv

    return total, dict(policy=policy_loss, value_ce=value_ce, value_q=value_q, adv=adv)

# ---------------- batch preparation ----------------
def get_batch(data, idx):
    """Slice one batch out of the tensors from load_selfplay_tensors (already on the device)."""
    return (data["x"][idx], data["legal"][idx], dense_targets(data, idx), data["played"][idx],
            data["vclass"][idx], data["q"][idx], data["sampled"][idx])


def collect_training_examples(version, db_path):
    """Build a training database at db_path from up to N_GAMES_SAMPLED games, sampled
    uniformly from the last WINDOW_SIZE self-play databases of this model version."""
    os.makedirs("temp_dbs", exist_ok=True)
    cons = {}
    for i in range(WINDOW_SIZE):
        src = f"self_play_records/records_{version[0]}_{version[1] - i}.db"
        if os.path.exists(src):
            shutil.copy2(src, f"temp_dbs/{i}.db")
            cons[i] = sqlite3.connect(f"temp_dbs/{i}.db")

    # every (db index, game id) in the window, then a uniform sample of them
    all_games = []
    for i in sorted(cons.keys()):
        all_games += [(i, row[0]) for row in cons[i].execute("SELECT id FROM self_play_games")]
    if len(all_games) > N_GAMES_SAMPLED:
        all_games = random.sample(all_games, N_GAMES_SAMPLED)

    con = setup_db(db_path=db_path)
    for i in sorted(cons.keys()):
        selected_game_ids = [game_id for db, game_id in all_games if db == i]
        if selected_game_ids:
            placeholders = ",".join("?" * len(selected_game_ids))
            # ids are offset per source db so they stay unique in the combined db
            games = cons[i].execute(f"SELECT * FROM self_play_games WHERE id IN ({placeholders})", selected_game_ids).fetchall()
            con.executemany(f"INSERT INTO self_play_games VALUES (? + {i * 10_000}, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", games)
            positions = cons[i].execute(f"SELECT * FROM self_play_positions WHERE game_id IN ({placeholders})", selected_game_ids).fetchall()
            con.executemany(f"INSERT INTO self_play_positions VALUES (? + {i * 100_000}, ? + {i * 10_000}, ?, ?, ?, ?, ?, ?)", positions)
        cons[i].close()
    con.commit()
    con.close()



# ---------------- main ----------------
def run_training(LOAD_PATH, SAVE_PATH, version):
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    torch.backends.cudnn.benchmark = True       # batch shape is fixed, so the autotuning pays off
    print("training on", device)

    db_path = "temp_training.db"
    collect_training_examples(version, db_path)

    train_ids, val_ids = split_game_ids(db_path, VAL_FRAC)
    train, train_meta = load_selfplay_tensors(db_path, train_ids, device)
    val, val_meta = load_selfplay_tensors(db_path, val_ids, device)
    print("train:", train_meta)
    print("val games:", val_meta["n_games"], "positions:", val_meta["n_positions"])
    komi = train_meta["komi"]
    assert val_meta["komi"] == komi
    assert train_meta["encoder_version"] == ENCODER_VERSION, "encoder version mismatch"
    n_train, n_val = train_meta["n_positions"], val_meta["n_positions"]

    info = torch.load(f"model_checkpoints/{LOAD_PATH}", map_location=device)
    params = info["params"]
    model = TakNet(**params)
    model.load_state_dict(info["weights"])
    model.to(device, memory_format=torch.channels_last)    # faster convs on tensor-core GPUs
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=DECAY)
    if 'opt' in info:
        optimizer.load_state_dict(info['opt'])
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    perms = [p.to(device) for p in build_perms()]

    best = float("inf")
    for epoch in range(EPOCHS):
        model.train()
        # running sums stay on the GPU; calling .item() every step would stall it
        sums, count = {}, 0
        order = torch.randperm(n_train, device=device)
        for start in range(0, n_train - BATCH + 1, BATCH):       # drop_last
            x, legal, target, played, vclass, q, sampled = get_batch(train, order[start:start + BATCH])
            x, legal, target, played = augment(x, legal, target, played, perms)
            x = x.contiguous(memory_format=torch.channels_last)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                policy, value = model(x)
            loss, parts = compute_losses(policy, value, legal, target, played, vclass, q, sampled)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            parts["total"] = loss
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + v.detach()
            count += 1
        print(f"epoch {epoch} train " + " ".join(f"{k} {v.item() / count:.4f}" for k, v in sums.items()))

        model.eval()
        vs, vcount = {}, 0
        with torch.no_grad():
            for start in range(0, n_val, BATCH):
                idx = torch.arange(start, min(start + BATCH, n_val), device=device)
                x, legal, target, played, vclass, q, sampled = get_batch(val, idx)
                x = x.contiguous(memory_format=torch.channels_last)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    policy, value = model(x)
                loss, parts = compute_losses(policy, value, legal, target, played, vclass, q, sampled)
                parts["total"] = loss
                logits = policy.float().flatten(1).masked_fill(~legal, -1e9)
                parts["policy_acc"] = (logits.argmax(1) == target.argmax(1)).float().mean()
                parts["value_acc"] = (value.argmax(1) == vclass).float().mean()
                for k, v in parts.items():
                    vs[k] = vs.get(k, 0.0) + v
                vcount += 1
        vs = {k: v.item() / vcount for k, v in vs.items()}
        print(f"epoch {epoch} val   " + " ".join(f"{k} {v:.4f}" for k, v in vs.items()))

        # select on the terms that are comparable across runs (not the adv term)
        score = vs["policy"] + W_V * vs["value_ce"]
        if score < best:
            best = score
            torch.save({"params": params, "weights": model.state_dict(),
                        "komi": komi, "encoder_version": ENCODER_VERSION,
                        "opt": optimizer.state_dict()},
                        f"model_checkpoints/{SAVE_PATH}")
            print("saved best")







#
#
# # Gate Model
#
# N_GATE_GAMES = 100
# GATE_KOMI = 2
# GATE_MAX_SIMS = 5_000
# GATE_N_PARALLEL = 64          # games played at the same time; their leaves share one GPU batch
# GATE_LEAVES_PER_GAME = 32     # leaves collected from each tree per round (was 256 from a single tree)
# GATE_C_PUCT = 1.5
# GATE_URGENCY = -0.2
# GATE_ALPHA = 0.05
# GATE_EPS = 0.25
# GATE_EVAL_CACHE = True        # reuse model evals for repeated / symmetric positions within a game
# GATE_CACHE_MAX_AGE = 2        # drop cache entries unused for this many moves
#
#
# class GateRunner:
#     """Plays many games at once and batches every game's leaves into one model call.
#
#     Speedups over playing one game at a time:
#       * GPU batches are large (up to N_PARALLEL * LEAVES_PER_GAME / 2 rows) and
#         full, instead of whatever was left of 256 after duplicate leaves were dropped.
#       * Games are split into two groups. While the GPU evaluates one group, the
#         CPU (Rust collect/submit) works on the other.
#       * collect_batch / submit_batch search every tree in a group in parallel,
#         one tree per CPU core, with the GIL released.
#       * The forward pass runs in float16 autocast with channels_last, and transfers
#         use pinned memory so they can overlap with other work.
#     """
#     def __init__(self, model_path_1, model_path_2, komi, max_sims, n_parallel, leaves_per_game,
#                  c_puct, urgency, alpha, eps):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.cuda = self.device.type == "cuda"
#
#         info = torch.load(model_path_1, map_location=self.device)
#         self.model_1 = TakNet(**info["params"])
#         self.model_1.load_state_dict(info["weights"])
#         self.model_1.to(self.device).eval()  # eval() matters: BatchNorm must not use batch stats
#         # channels_last makes the convs about 1.5x faster on tensor-core GPUs
#         self.model_1.to(memory_format=torch.channels_last)
#
#         info = torch.load(model_path_2, map_location=self.device)
#         self.model_2 = TakNet(**info["params"])
#         self.model_2.load_state_dict(info["weights"])
#         self.model_2.to(self.device).eval()  # eval() matters: BatchNorm must not use batch stats
#         # channels_last makes the convs about 1.5x faster on tensor-core GPUs
#         self.model_2.to(memory_format=torch.channels_last)
#
#         self.model_path_1 = model_path_1
#         self.model_path_2 = model_path_2
#
#         self.komi = komi
#         self.max_sims = max_sims
#         self.n_parallel = n_parallel
#         self.leaves = leaves_per_game
#         self.c_puct = c_puct
#         self.urgency = urgency
#         self.alpha = alpha
#         self.eps = eps
#
#
#         per_group = (n_parallel + 1) // 2
#         self.groups = [Group(per_group * leaves_per_game, self.cuda) for _ in range(2)]
#         self.per_group = per_group
#
#     # ---- per game per model ----
#     def new_search(self, slot):
#         slot.search = tak_engine.Search(slot.game.to_tps(), urgency=self.urgency,
#                                         c_puct=self.c_puct, komi=self.komi,
#                                         eval_cache=GATE_EVAL_CACHE, cache_max_age=GATE_CACHE_MAX_AGE)
#         slot.noised = False
#
#     def add_noise(self, slot):
#         """Noise goes on the root's children, so the root must already be expanded."""
#         slot.search.add_noise(self.alpha, self.eps, slot.seed + slot.ply)
#         slot.noised = True
#
#     def play_move(self, slot, game):
#         """Search is done: pick and play a move. Returns an end reason, or None if the game goes on.
#
#         When the game goes on, the search is rebased onto the move played, so the
#         visits already spent below that move carry over to the next search.
#         """
#         s = slot.search
#         visit_counts = [(x[0], x[1]) for x in s.root_stats()]
#         mv = max(visit_counts, key=lambda x: x[1])[0]
#         slot.records.append((slot.game.to_tps(), slot.ply, mv, visit_counts, s.root_q()))
#         slot.game.make_move_raw(mv)
#         game.make_move_raw(mv)
#         slot.ply += 1
#         if slot.game.result() != 0:
#             return 'game_over'
#         s.rebase_tree(mv)
#         slot.noised = False
#         if s.sims() > 0:          # new root was already expanded: noise can go on right away
#             self.add_noise(slot)
#         return None
#
#     # ---- per group ----
#     def collect(self, group):
#         """CPU: every game in the group descends its tree and writes leaves into the group's buffer.
#
#         The trees are searched in parallel in Rust; rows come back packed in slot order.
#         """
#         group.spans = []
#         if group.slots:
#             counts = tak_engine.collect_batch([slot.search for slot in group.slots],
#                                               group.inputs_flat, self.leaves)
#             off = 0
#             for slot, n in zip(group.slots, counts):
#                 group.spans.append((slot, off, n))
#                 off += n
#         group.collected = bool(group.spans)
#         return sum(n for _, _, n in group.spans)
#
#     @torch.no_grad()
#     def launch(self, group, n):
#         """GPU: queue the model call for n rows. Returns without waiting for the result."""
#         x = group.inputs[:n].to(self.device, non_blocking=True).contiguous(memory_format=torch.channels_last)
#         with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.cuda):
#             policy, value = self.model(x)
#         pol = torch.softmax(policy.float().flatten(1), dim=1)
#         vp = torch.softmax(value.float(), dim=1)
#         group.policy[:n].copy_(pol, non_blocking=True)
#         group.values[:n].copy_(vp[:, 2] - vp[:, 0], non_blocking=True)
#         if self.cuda:
#             group.event = torch.cuda.Event()
#             group.event.record()
#
#     def finish(self, group, start_game, write_game):
#         """Wait for the group's results, hand them to each search, and advance finished searches."""
#         if not group.collected:
#             return
#         if group.event is not None:
#             group.event.synchronize()
#             group.event = None
#         total = sum(n for _, _, n in group.spans)
#         tak_engine.submit_batch([slot.search for slot, _, _ in group.spans],
#                                 group.policy_flat[:total * POLICY_LEN], group.values_np[:total])
#         kept = []
#         for slot, off, n in group.spans:
#             s = slot.search
#             if not slot.noised and s.sims() > 0:   # first submit at this root expanded it
#                 self.add_noise(slot)
#             if s.sims() >= self.max_sims:
#                 end_reason = self.play_move(slot)
#                 if end_reason is not None:
#                     write_game(slot, end_reason)
#                     slot = start_game()
#                     if slot is None:
#                         continue
#             kept.append(slot)
#         group.slots = kept
#         group.spans = []
#         group.collected = False
#
#     def run(self, num_games, conn, save_path):
#         rand = np.random.default_rng()
#         started = 0
#         finished = 0
#
#         def start_game():
#             nonlocal started
#             if started >= num_games:
#                 return None
#             started += 1
#             slot = Slot(self.komi, int(rand.integers(0, 1 << 63)))
#             self.new_search(slot)
#             return slot
#
#         def write_game(slot, end_reason):
#             nonlocal finished
#             game_id = conn.execute('''
#                 INSERT INTO self_play_games (
#                     seed, model, encoder_version, komi, max_sims, batch_size, c_puct,
#                     urgency, alpha, eps, temp_plies, n_plies, outcome, end_reason
#                 )
#                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
#             ''', (
#                 slot.seed, self.model_path, ENCODER_VERSION, self.komi, self.max_sims, self.leaves,
#                 self.c_puct, self.urgency, self.alpha, self.eps, self.temp_plies,
#                 slot.ply, slot.game.result(), end_reason
#             )).lastrowid
#             # record = (tps, ply, mv, visit_counts, q)
#             conn.executemany('''
#                 INSERT INTO self_play_positions (game_id, ply, tps, move_raw, sampled, q, visits)
#                 VALUES (?, ?, ?, ?, ?, ?, ?)
#             ''', [
#                 (game_id, ply, tps, mv, 1 if ply < self.temp_plies else 0, q, pack_visits(visits))
#                 for tps, ply, mv, visits, q in slot.records
#             ])
#             conn.commit()
#             finished += 1
#             print(f'Game {game_id} written to database ({slot.ply} plies, {end_reason}).')
#             if finished % 10 == 0:
#                 copy_db(conn, f'self_play_records/{save_path}')
#
#         for i in range(self.n_parallel):
#             slot = start_game()
#             if slot is None:
#                 break
#             self.groups[i % 2].slots.append(slot)
#
#         # pipeline: collect A, launch A, finish B, collect B, launch B, finish A, ...
#         # the GPU works on one group while the CPU finishes and collects the other
#         while any(g.slots or g.collected for g in self.groups):
#             for g, other in ((0, 1), (1, 0)):
#                 group = self.groups[g]
#                 n = self.collect(group)
#                 if n:
#                     self.launch(group, n)
#                 self.finish(self.groups[other], start_game, write_game)
#
#         copy_db(conn, f'self_play_records/{save_path}')
#
# def gate_model(base_version, new_version):
#     pass
#












# iterate between self play and training, save checkpoints to drive periodically

# set these to the newest model on drive before (re)starting the loop
model_version = 13
model_sub_version = 10
while True:
    model_path = f'version_{model_version}_{model_sub_version}.pt'
    model_save_path = f'version_{model_version}_{model_sub_version + 1}.pt'

    db_path = f'records_{model_version}_{model_sub_version}.db'

    run_self_play(model_path, db_path)
    run_training(model_path, model_save_path, (model_version, model_sub_version))
    model_sub_version += 1

