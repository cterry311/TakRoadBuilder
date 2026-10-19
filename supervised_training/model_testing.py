import tak_engine
from tak_engine import Tak
from model.tak_net import TakNet
from supervised_training.build_table import test_positions
from supervised_training.training_example_loader import TakDataset, DataLoader
from torch.nn import functional as F
import torch
import numpy as np


test_loader = DataLoader(TakDataset(test_positions), batch_size=128, shuffle=True)

def test_model(model_path, value_weight=0.5):
    info = torch.load(f'model_checkpoints/{model_path}')
    model = TakNet(**info['params'])
    model.load_state_dict(info['weights'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    inputs = torch.empty(128, 25, 6, 6)
    inputs_np = inputs.numpy()
    output_masks = torch.empty(128, 251, 6, 6, dtype=torch.uint8)
    output_masks_np = output_masks.numpy()

    losses = []
    policy_losses = []
    value_losses = []

    policy_accuracy = []
    value_accuracy = []

    top_move_in_best_class = []
    top_moves_in_best_class = []

    with torch.no_grad():
        for result, tps, move in test_loader:
            n = len(result)
            tak_engine.encode_training_batch(tps, inputs_np[:n], output_masks_np[:n])

            tak_games: list[Tak] = tak_engine.from_tps_bulk(tps)
            filtered_moves = [x.filtered_moves_raw(1, False)[0] for x in tak_games]
            filtered_move_count = [len(x) for x in filtered_moves]

            value_target = (result + 1).to(device)
            policy_target = torch.tensor(
                tak_engine.bulk_raw_to_index(move.tolist()), device=device
            )

            x = inputs[:n].to(device)
            legal = output_masks[:n].reshape(n, -1).to(device).bool()
            policy, value = model(x)
            policy_logits = policy.flatten(1).masked_fill(~legal, -1e9)
            policy_probs = F.softmax(policy_logits, dim=1)
            index_order = torch.argsort(policy_probs, dim=1, descending=True)

            policy_loss = F.cross_entropy(policy_logits, policy_target)
            value_loss = value_weight * F.cross_entropy(value, value_target)
            loss = policy_loss + value_loss

            value_acc = (value.argmax(1) == value_target).float().mean().item()
            policy_acc = (policy_logits.argmax(1) == policy_target).float().mean().item()

            top_move_in_filtered_moves = sum([filtered_moves[i].count(tak_engine.index_to_raw(index_order[i][0].item())) > 0 for i in range(n)]) / n
            top_moves_in_filtered_moves_ratio = sum([sum([filtered_moves[n_i].count(tak_engine.index_to_raw(x[j].item())) > 0 for j in range(i)]) / i for x, i, n_i in zip(index_order, filtered_move_count, range(n))]) / n

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            losses.append(loss.item())

            value_accuracy.append(value_acc)
            policy_accuracy.append(policy_acc)

            top_move_in_best_class.append(top_move_in_filtered_moves)
            top_moves_in_best_class.append(top_moves_in_filtered_moves_ratio)

    avg_loss = sum(losses) / len(losses)
    avg_policy_loss = sum(policy_losses) / len(policy_losses)
    avg_value_loss = sum(value_losses) / len(value_losses)

    avg_policy_acc = sum(policy_accuracy) / len(policy_accuracy)
    avg_value_acc = sum(value_accuracy) / len(value_accuracy)

    avg_top_move_in_best_class = sum(top_move_in_best_class) / len(top_move_in_best_class)
    avg_top_moves_in_best_class = sum(top_moves_in_best_class) / len(top_moves_in_best_class)

    print('-------loss-----------')
    print(f'avg loss {avg_loss}')
    print(f'avg policy loss {avg_policy_loss}')
    print(f'avg value loss {avg_value_loss}')
    print('-------accuracy-----------')
    print(f'avg policy acc {avg_policy_acc}')
    print(f'avg value acc {avg_value_acc}')
    print('-------top move analisis-------------')
    print(f'avg top move in best class {avg_top_move_in_best_class}')
    print(f'avg top moves in best class {avg_top_moves_in_best_class}')


if __name__ == '__main__':
    test_model('test3.pt')