import tak_engine
from sympy import Float

from model.tak_net import TakNet
from supervised_training.build_table import train_positions, val_positions, test_positions
from supervised_training.training_example_loader import TakDataset, DataLoader
from torch.nn import functional as F
import torch
import numpy as np

train_loader = DataLoader(TakDataset(train_positions), batch_size=128, shuffle=True)
val_loader = DataLoader(TakDataset(val_positions), batch_size=128, shuffle=True)
test_loader = DataLoader(TakDataset(test_positions), batch_size=128, shuffle=True)




# direction order matches the engine: 0 '+', 1 '-', 2 '<', 3 '>' as (dy, dx)
DIRS = [(1, 0), (-1, 0), (0, -1), (0, 1)]
TRANSFORMS = [(k, flip) for flip in (False, True) for k in range(4)]   # 8 total

def apply_T(t, k, flip):
    t = torch.rot90(t, k, dims=(-2, -1))
    return torch.flip(t, dims=(-1,)) if flip else t

def build_perms():
    base = torch.arange(36).reshape(6, 6)
    perms = []
    for k, flip in TRANSFORMS:
        moved = apply_T(base, k, flip).flatten()       # moved[new_cell] = old_cell
        new_of_old = torch.empty(36, dtype=torch.long)
        new_of_old[moved] = torch.arange(36)

        a = divmod(int(new_of_old[2 * 6 + 2]), 6)      # where a probe cell lands
        dir_map = []
        for dy, dx in DIRS:
            b = divmod(int(new_of_old[(2 + dy) * 6 + 2 + dx]), 6)
            dir_map.append(DIRS.index((b[0] - a[0], b[1] - a[1])))

        perm = torch.empty(251 * 36, dtype=torch.long)  # old index -> new index
        for ch in range(251):
            if ch < 3:
                nch = ch                                # placements keep their channel
            else:
                s = ch - 3
                nch = 3 + (s // 4) * 4 + dir_map[s % 4] # mask part unchanged, dir remapped
            for cell in range(36):
                perm[ch * 36 + cell] = nch * 36 + int(new_of_old[cell])
        perms.append(perm)
    return perms

PERMS = build_perms()

def augment(x, legal, target, perms):
    # x: (n,25,6,6)  legal: (n,9036) bool  target: (n,) int64, all on the same device
    n = x.shape[0]
    ts = torch.randint(0, 8, (n,), device=x.device)
    x_out, legal_out, target_out = x.clone(), torch.empty_like(legal), target.clone()
    for t, (k, flip) in enumerate(TRANSFORMS):
        idx = (ts == t).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        perm = perms[t]
        x_out[idx] = apply_T(x[idx], k, flip)
        legal_out[idx[:, None], perm[None, :]] = legal[idx]   # legal_new[perm[i]] = legal[i]
        target_out[idx] = perm[target[idx]]
    return x_out, legal_out, target_out


def run_training(model_load_path, model_save_path, EPOCHS=100, value_weight=0.5, params=None):
    if params is None:
        params = {}
    if model_load_path is None:
        model = TakNet(**params)
    else:
        info = torch.load(f'model_checkpoints/{model_load_path}')
        model = TakNet(**info['params'])
        params = info['params']
        model.load_state_dict(info['weights'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f'training on {device}')

    PERMS_DEV = [p.to(device) for p in PERMS]

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    inputs = torch.empty(128, 25, 6, 6)
    inputs_np = inputs.numpy()
    output_masks = torch.empty(128, 251, 6, 6, dtype=torch.uint8)
    output_masks_np = output_masks.numpy()

    best_loss = Float('inf')

    for epoch in range(EPOCHS):
        model.train()
        for result, tps, move in train_loader:
            n = len(result)
            tak_engine.encode_training_batch(tps, inputs_np[:n], output_masks_np[:n])

            # x = inputs[:n]
            # legal = output_masks[:n].reshape(n, -1).bool()  # (n, 9036)
            # policy_target = torch.tensor(
            #     tak_engine.bulk_raw_to_index(move.tolist()))  # (n,) int64
            #
            # x, legal, policy_target = augment(x, legal, policy_target, PERMS)
            # x = x.to(device)
            # legal = legal.to(device)
            # policy_target = policy_target.to(device)

            x = inputs[:n].to(device)
            legal = output_masks[:n].reshape(n, -1).bool().to(device)
            policy_target = torch.tensor(tak_engine.bulk_raw_to_index(move.tolist())).to(device)
            x, legal, policy_target = augment(x, legal, policy_target, PERMS_DEV)

            value_target = (result + 1).to(device)  # (n,) int64

            optimizer.zero_grad()
            policy, value = model(x)
            policy_logits = policy.flatten(1).masked_fill(~legal, -1e9)

            loss = F.cross_entropy(policy_logits, policy_target) + value_weight * F.cross_entropy(value, value_target)
            loss.backward()
            optimizer.step()
        model.eval()
        total_loss = []
        policy_loss = []
        policy_acc = []
        value_loss = []
        value_acc = []
        for result, tps, move in val_loader:
            n = len(result)
            tak_engine.encode_training_batch(tps, inputs_np[:n], output_masks_np[:n])

            x = inputs[:n].to(device)
            legal = output_masks[:n].reshape(n, -1).to(device).bool()
            policy_target = torch.tensor(
                tak_engine.bulk_raw_to_index(move.tolist()), device=device
            )
            value_target = (result + 1).to(device)
            policy, value = model(x)
            policy_logits = policy.flatten(1).masked_fill(~legal, -1e9)
            p_loss = F.cross_entropy(policy_logits, policy_target)
            v_loss = value_weight * F.cross_entropy(value, value_target)
            loss = p_loss + v_loss
            total_loss.append(loss.item())
            policy_loss.append(p_loss.item())
            value_loss.append(v_loss.item())
            policy_acc.append((policy_logits.argmax(1) == policy_target).float().mean().item())
            value_acc.append((value.argmax(1) == value_target).float().mean().item())
        avg_loss = np.mean(total_loss)
        avg_policy_loss = np.mean(policy_loss)
        avg_value_loss = np.mean(value_loss)
        avg_policy_acc = np.mean(policy_acc)
        avg_value_acc = np.mean(value_acc)
        print(f'epoch {epoch} loss {avg_loss:.4f} policy_loss {avg_policy_loss:.4f} value_loss {avg_value_loss:.4f} policy_acc {avg_policy_acc:.4f} value_acc {avg_value_acc:.4f}')
        if avg_loss < best_loss:
            best_loss = avg_loss
            print('new best loss')
            info = {
                'params': params,
                'weights': model.state_dict(),
            }
            torch.save(info, f"model_checkpoints/{model_save_path}")



if __name__ == '__main__':
    run_training('version_6_1.pt', 'version_6_2.pt', EPOCHS=200)