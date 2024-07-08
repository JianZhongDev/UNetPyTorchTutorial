"""
FILENAME: ResidualLayers.py
DESCRIPTION: Residual connection layers
@author: Jian Zhong
"""

import torch
from torch import nn
from torch import Tensor
from typing import Optional


# addtional residual layer
class ResdiualAdd(nn.Module):
    def __init__(self, block: nn.Module, shortcut: Optional[nn.Module] = None):
        super().__init__()
        self.block = block
        self.shortcut = shortcut

    def forward(self, x: Tensor) -> Tensor:
        res = x
        x = self.block(x)
        if self.shortcut:
            res = self.shortcut(res)
        x += res
        return x
    

# concatenate residual layer
class ResdiualCat(nn.Module):
    def __init__(self, block: nn.Module, shortcut: Optional[nn.Module] = None):
        super().__init__()
        self.block = block
        self.shortcut = shortcut

    def forward(self, x: Tensor) -> Tensor:
        res = x
        x = self.block(x)
        if self.shortcut:
            res = self.shortcut(res)
        y = torch.cat([x, res], dim = 1)
        return y