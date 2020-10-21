# author: yx
# date: 2020/10/16 13:50
import numpy as np

mapping = {'cp_type':{'trt_cp': 1, 'ctl_vehicle': 2},
           'cp_tine':{48: 1, 72: 2, 24: 3},
           'cp_dose':{'D1': 1, 'D2': 2}}


def transform_data(train, test, col, mapping, normalize=True):
    """
        the first 3 columns represents categories, the others numericals features
    """

    categories_tr = np.stack([train[c].apply(lambda x: mapping[c][x]).values for c in col[:3]], axis=1)
    categories_test = np.stack([test[c].apply(lambda x: mapping[c][x]).values for c in col[:3]], axis=1)

    max_ = 10.
    min_ = -10.

    numerical_tr = train[col[3:]].values
    numerical_test = test[col[3:]].values

    if normalize:
        numerical_tr = (numerical_tr - min_) / (max_ - min_)
        numerical_test = (numerical_test - min_) / (max_ - min_)
    return categories_tr, categories_test, numerical_tr, numerical_test