# author: yx
# date: 2020/10/16 11:35
import numpy as np
import pandas as pd
import torch as t
from torch.utils import data
from ..transforms.transforms import transform

# transform map
mapping = {'cp_type': {'trt_cp': 1, 'ctl_vehicle': 2},
           'cp_time': {48: 1, 72: 2, 24: 3},
           'cp_dose': {'D1': 1, 'D2': 2}}


def get_data_rows_num(cfg):
    """
    get data rows num, so we can use k-fold
    :param cfg: config file
    :return: rows number
    """
    train_dir = cfg.DATASET.TRAIN
    train = pd.read_csv(train_dir)
    return len(train)


def get_ratio_labels(df):
    columns = list(df.columns)
    columns.pop(0)
    ratios = []
    toremove = []
    for c in columns:
        counts = df[c].value_counts()
        if len(counts) != 1:
            ratios.append(counts[0] / counts[1])
        else:
            toremove.append(c)
    print(f"remove {len(toremove)} columns")

    for t in toremove:
        columns.remove(t)
    return columns


# def transform_data(features, normalize=True):
#     """
#         the first 3 columns represents categories, the others numericals features
#     """
#     max_ = 10.
#     min_ = -10.
#
#     col = list(features.columns)[1:]
#     sig = list(features.columns)[0]
#
#     # sig_ids = np.stack([features[sig].values], axis=1)
#
#     categories = np.stack([features[c].apply(lambda x: mapping[c][x]).values for c in col[:3]], axis=1)
#
#     numerical = features[col[3:]].values
#
#     if normalize:
#         numerical = (numerical - min_) / (max_ - min_)
#     return categories, numerical

def transform_data(scale, features):
    scale = scale

    col = list(features.columns)[1:]
    categories = np.stack([features[c].apply(lambda x: mapping[c][x]).values for c in col[:3]], axis=1)

    numerical = transform(scale, features)
    return categories, numerical
    pass


def data_preprocess(cfg, indices=None, is_train=True):
    """
    load data, transform data and change data type
    :param cfg: config file
    :param indices: indices by k-fold
    :param is_train: is train or test
    :return: data content
    """
    normalize = cfg.DATASET.NORMALIZE
    remove_vehicle = cfg.DATASET.REMOVE_VEHICLE
    scale = cfg.DATASET.SCALE

    if is_train:
        train_dir = cfg.DATASET.TRAIN
        train_targets_scored_dir = cfg.DATASET.TRAIN_TARGETS_SCORED
        train_targets_non_scored_dir = cfg.DATASET.TRAIN_TARGETS_NON_SCORED

        train = pd.read_csv(train_dir)
        train_targets_scored = pd.read_csv(train_targets_scored_dir)
        train_targets_non_scored = pd.read_csv(train_targets_non_scored_dir)

        # column 'sig_id' data is useless, so we drop out this column
        train_targets_scored = train_targets_scored.drop(['sig_id'], axis=1)
        train_targets_non_scored = train_targets_non_scored.drop(['sig_id'], axis=1)

        # remove highly imbalanced labels in nonscored
        columns_non_scored = get_ratio_labels(train_targets_non_scored)

        # select data by k-fold indices
        if indices is not None:
            train = train.loc[indices]
            train_targets_scored = train_targets_scored.loc[indices]
            train_targets_non_scored = train_targets_non_scored.loc[indices]

            if remove_vehicle:
                train_targets_scored = train_targets_scored.loc[train['cp_type']=='trt_cp'].reset_index(drop=True)
                train_targets_non_scored = train_targets_non_scored.loc[train['cp_type'] == 'trt_cp'].reset_index(drop=True)
                train = train.loc[train['cp_type'] == 'trt_cp'].reset_index(drop=True)

        assert len(train) == len(train_targets_scored) and len(train_targets_scored) == len(train_targets_non_scored)

        # categories, numerical = transform_data(train, normalize=normalize)
        categories, numerical = transform_data(scale, train)

        # type change
        train_targets_scored = train_targets_scored[train_targets_scored.columns].values.astype(np.float32)

        train_targets_non_scored = train_targets_non_scored[columns_non_scored].values.astype(np.float32)

        return categories, numerical, train_targets_scored, train_targets_non_scored
    else:
        test_dir = cfg.DATASET.TEST
        test = pd.read_csv(test_dir)
        categories, numerical = transform_data(scale, test)
        return categories, numerical


class MOADataset(data.Dataset):
    def __init__(self, cfg, transforms=None, indices=None, is_train=True):
        self.cfg = cfg
        self.transforms = transforms
        self.indices = indices
        self.is_train = is_train
        if is_train:
            self.cats, self.nums, self.y, self.y2 = data_preprocess(self.cfg, indices=self.indices, is_train=is_train)
            pass
        else:
            self.cats, self.nums = data_preprocess(self.cfg, indices=self.indices, is_train=is_train)
            pass

    def __len__(self):
        return len(self.cats)

    def __getitem__(self, index):
        x1 = t.as_tensor(self.cats[index], dtype=t.long)
        x2 = t.as_tensor(self.nums[index], dtype=t.float)

        if self.is_train:
            label = t.as_tensor(self.y[index], dtype=t.float)
            label2 = t.as_tensor(self.y2[index], dtype=t.float)
            return x1, x2, label, label2
        else:
            return x1, x2
