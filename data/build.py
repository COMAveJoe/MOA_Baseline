# author: yx
# date: 2020/10/16
from .datasets.LishMoa import MOADataset
from torch.utils import data


def build_dataset(cdg, indices=None, is_train=True):
    datasets = MOADataset(cfg=cdg, indices=indices, is_train=is_train)
    return datasets


def make_data_loader(cfg, indices=None, is_train=True):
    if is_train:
        batch_size = cfg.SOLVER.BATCH_SIZE
        pass
    else:
        batch_size = cfg.TEST.BATCH_SIZE
        pass

    datasets = build_dataset(cfg, indices, is_train)

    data_loader = data.DataLoader(
        datasets, batch_size=batch_size, pin_memory=True)

    return data_loader
