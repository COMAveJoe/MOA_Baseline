# author: yx
# date: 2020/10/16 17:17
import torch
from .MOA_MLP import MOA_MLP, MOA_MLPv2

_META_ARCHITRCTURE = {'MOA_MLP': MOA_MLP, 'MOA_MLPv2': MOA_MLPv2}


def build_model(cfg):
    meta_arch = _META_ARCHITRCTURE[cfg.MODEL.META_ARCHITECTURE]
    model = meta_arch(cfg)
    return model
