"""
FILENAME: LossFunctions.py
DESCRIPTION: Loss function definitions for classification
@author: Jian Zhong
"""

import torch
from torch import nn


## Cross entropy loss with spatial weight
class WeightedCrossEntropyLoss(nn.Module):

    def __init__(
            self, 
            reduction = "mean",
    ):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target, weight):
        
        batch_size = pred.size(0)
        nof_classes = pred.size(1)

        pk = torch.exp(pred)
        pk = pk / torch.sum(pk, dim = 1, keepdims = True)
        
        cross_entropy = None
        for i_class in range(nof_classes):
            cur_class = i_class
            cur_target_prob = target == cur_class
            cur_pk = pk[:, i_class, ...]
            cur_cross_entropy = torch.mean(weight * torch.log(cur_pk) * cur_target_prob)
            if cross_entropy is None:
                cross_entropy = cur_cross_entropy
            else:
                cross_entropy += cur_cross_entropy

        loss = - cross_entropy
        if self.reduction != "mean":
            loss = loss * batch_size

        return loss


## Cross entropy loss 
class CrossEntropyLoss(nn.Module):

    def __init__(
            self, 
            reduction = "mean",
    ):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        
        batch_size = pred.size(0)
        nof_classes = pred.size(1)

        pk = torch.exp(pred)
        pk = pk / torch.sum(pk, dim = 1, keepdims = True)
        
        cross_entropy = None
        for i_class in range(nof_classes):
            cur_class = i_class
            cur_target_prob = (target == cur_class).to(float)
            cur_pk = pk[:, i_class, ...]
            cur_cross_entropy = torch.mean(torch.log(cur_pk) * cur_target_prob)
            if cross_entropy is None:
                cross_entropy = cur_cross_entropy
            else:
                cross_entropy += cur_cross_entropy

        loss = - cross_entropy
        if self.reduction != "mean":
            loss = loss * batch_size
    
        return loss



