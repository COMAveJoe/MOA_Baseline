# author: yx
# date: 2020/10/16 17:12

import torch


def make_optimizer(cfg, model):
    params = []
    optimizer = None
    lr = 0

    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue

        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = torch.optim.SGD(params, lr, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.OPTIMIZER_NAME == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr)
    return optimizer
