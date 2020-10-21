# author: yx
# date: 2020/10/16 16:58

from torch import nn


def bce_with_logits_loss(input, target):
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    loss = criterion(input, target)
    return loss