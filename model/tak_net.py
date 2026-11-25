"""
defining model structure
"""
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

IN_CHANNELS = 6 + (2 * 6) + 1 + 1 + 2 + 2 + 1 # 25

TRUNK_CHANNELS = 64
TRUNK_BLOCKS = 6

POLICY_OUTPUT_CHANNELS = 251
POLICY_BLOCKS = 1

VALUE_NET_SIZE = 64
VALUE_BLOCKS = 1



class ResBlock(nn.Module):
    """
    a conv block with a connection between the input and output
    """
    def __init__(self, ch):
        """
        create the block
        :param ch: the number of channels in the block
        """
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x) -> torch.Tensor:
        """
        do a forward pass through the block
        :param x: the input
        :return: the output
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


class TakNet(nn.Module):
    """
    an alphazero-style network
    """
    def __init__(self, in_ch: int=IN_CHANNELS, tr_ch: int=TRUNK_CHANNELS, tr_bl: int=TRUNK_BLOCKS, pol_ch: int=POLICY_OUTPUT_CHANNELS, pol_bl: int=POLICY_BLOCKS, val_size: int=VALUE_NET_SIZE, val_bl: int=VALUE_BLOCKS,):
        """
        create the network
        :param in_ch: the number of input channels, constrained by model input
        :param tr_ch: the number of channels to be used in the trunk
        :param tr_bl: the number of resblocks in the trunk
        :param pol_ch: the number of channels to be used in the policy head
        :param pol_bl: the number of resblocks in the policy head
        :param val_size: the width of the value head
        :param val_bl: the number of layers in the value head
        """
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

    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        """
        do a forward pass through the network
        :param x: the input
        :return: a tuple of (policy output, value output)
        """
        x = self.trunk(x)
        return self.policy(x), self.value(x) # value output, 0 = loss, 1 = draw, 2 = win
