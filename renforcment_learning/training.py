import torch
import numpy as np
import tak_engine
from torch.nn import functional as F
from torch.utils.data import DataLoader

from model.tak_net import TakNet
from renforcment_learning.reinforcment_data_loader import SelfPlayDataset, split_game_ids

from engine_clients.tiltak_bot import Tiltak

# ---------------- settings ----------------
DB_PATH = "self_play_records/self_play_1.db"
LOAD_PATH = "version_9_1.pt"          # inside model_checkpoints/
SAVE_PATH = "version_9_5.pt"
EPOCHS = 20
BATCH = 128
LR = 3e-4                             # lower than supervised (guess, tune)
W_V = 0.5                             # value cross-entropy on game outcome
GAMMA = 0.25                          # value vs search q
BETA = 0.1                            # advantage term; set 0 to disable
ADV_CLIP = 1.0
VAL_FRAC = 0.05
ENCODER_VERSION = 2                   # must match what the data was generated with

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
def compute_losses(policy, value_logits, legal, target, played, value_class, q, sampled, tiltak_values=None):
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

    if tiltak_values is not None:
        # tiltak_losses = (F.mse_loss(probs[:, 2] - probs[:, 0], tiltak_values, reduction='none') * sampled).mean() # sampled moves only
        tiltak_losses = F.mse_loss(probs[:, 2] - probs[:, 0], tiltak_values)   # all moves
    else:
        tiltak_losses = torch.scalar_tensor(0)

    total = policy_loss + W_V * value_ce + GAMMA * value_q + BETA * adv + tiltak_losses * 2.0

    return total, dict(policy=policy_loss, value_ce=value_ce, value_q=value_q, adv=adv, tiltak_losses=tiltak_losses)

# ---------------- batch preparation ----------------
def prepare(batch, bufs, komi, device):
    tps, target, played, value_class, q, sampled = batch
    n = len(tps)
    inputs, inputs_np, masks, masks_np = bufs
    tak_engine.encode_training_batch(tps, inputs_np[:n], masks_np[:n], komi=komi)
    x = inputs[:n].to(device)
    legal = masks[:n].reshape(n, -1).to(device).bool()
    return (x, legal, target.to(device), played.to(device).long(),
            value_class.to(device).long(), q.to(device).float(), sampled.to(device).float())

# ----------------- tiltak evals ---------------

_tiltak = Tiltak(half_komi=4)

def get_tiltak_evals(tps: list[str], device):
    n = len(tps)
    values = torch.zeros(n)
    for i in range(n):
        values[i] = _tiltak.evaluate(position=tps[i], nodes=100)["value"]
    return values.to(device)



# ---------------- main ----------------
def run():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("training on", device)

    train_ids, val_ids = split_game_ids(DB_PATH, VAL_FRAC)
    train_ds = SelfPlayDataset(DB_PATH, train_ids)
    val_ds = SelfPlayDataset(DB_PATH, val_ids)
    print("train:", train_ds.metadata)
    print("val games:", val_ds.metadata["n_games"], "positions:", len(val_ds))
    komi = train_ds.komi
    assert val_ds.komi == komi
    assert train_ds.metadata["encoder_version"] == ENCODER_VERSION, "encoder version mismatch"

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False)

    info = torch.load(f"model_checkpoints/{LOAD_PATH}", map_location=device)
    params = info["params"]
    model = TakNet(**params)
    model.load_state_dict(info["weights"])
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    perms = [p.to(device) for p in build_perms()]
    inputs = torch.empty(BATCH, 25, 6, 6)
    masks = torch.empty(BATCH, 251, 6, 6, dtype=torch.uint8)
    bufs = (inputs, inputs.numpy(), masks, masks.numpy())

    best = float("inf")
    for epoch in range(EPOCHS):
        model.train()
        sums, count = {}, 0
        for batch in train_loader:
            tiltak_values = get_tiltak_evals(batch[0], device)
            x, legal, target, played, vclass, q, sampled = prepare(batch, bufs, komi, device)
            x, legal, target, played = augment(x, legal, target, played, perms)

            optimizer.zero_grad()
            policy, value = model(x)
            loss, parts = compute_losses(policy, value, legal, target, played, vclass, q, sampled, tiltak_values=tiltak_values)
            loss.backward()
            optimizer.step()

            parts["total"] = loss
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + v.item()
            count += 1
        print(f"epoch {epoch} train " + " ".join(f"{k} {v / count:.4f}" for k, v in sums.items()))

        model.eval()
        vs, vcount, pol_acc, val_acc = {}, 0, 0.0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                tiltak_values = get_tiltak_evals(batch[0], device)
                x, legal, target, played, vclass, q, sampled = prepare(batch, bufs, komi, device)
                policy, value = model(x)
                loss, parts = compute_losses(policy, value, legal, target, played, vclass, q, sampled, tiltak_values=tiltak_values)
                parts["total"] = loss
                for k, v in parts.items():
                    vs[k] = vs.get(k, 0.0) + v.item()
                logits = policy.flatten(1).masked_fill(~legal, -1e9)
                pol_acc += (logits.argmax(1) == target.argmax(1)).float().mean().item()
                val_acc += (value.argmax(1) == vclass).float().mean().item()
                vcount += 1
        print(f"epoch {epoch} val   " + " ".join(f"{k} {v / vcount:.4f}" for k, v in vs.items())
              + f" policy_acc {pol_acc / vcount:.4f} value_acc {val_acc / vcount:.4f}")

        # select on the terms that are comparable across runs (not the adv term)
        score = (vs["policy"] + W_V * vs["value_ce"]) / vcount
        if score < best:
            best = score
            torch.save({"params": params, "weights": model.state_dict(),
                        "komi": komi, "encoder_version": ENCODER_VERSION},
                       f"model_checkpoints/{SAVE_PATH}")
            print("saved best")

if __name__ == "__main__":
    print('starting')
    run()