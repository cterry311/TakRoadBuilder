from typing import List

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
import torch
import torch.nn as nn
import torch.nn.functional as F

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
            nn.BatchNorm1d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * 6 * 6, val_size),
            nn.ReLU(),
        ]
        for _ in range(val_bl):
            value_ops.append(nn.Linear(val_size, val_size))
            value_ops.append(nn.ReLU())
        value_ops.append(nn.Linear(val_size, 3))
        self.value = nn.Sequential(*value_ops)

    def forward(self, x):
        x = self.trunk(x)
        return self.policy(x), self.value(x)
