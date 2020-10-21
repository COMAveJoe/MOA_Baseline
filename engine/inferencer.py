# author: yx
# date: 2020/10/19 9:20

import torch
import os
from tqdm import trange
from data.datasets.LishMoa import data_preprocess

import numpy as np
import pandas as pd

def do_inference(
        cfg,
        model
):
    model.eval()

    infer_result_dir = os.path.join(cfg.INFERENCE_RESULT_DIR, 'submission.csv')
    submission_file = cfg.DATASET.SUBMISSION
    categories, numerical = data_preprocess(cfg, indices=None, is_train=False)
    test_features = pd.read_csv(cfg.DATASET.TEST)

    sub = pd.read_csv(submission_file)
    p_min = 0.001
    p_max = 0.999

    for i in trange(len(categories)):
        cat = torch.tensor([categories[i]])
        num = torch.tensor([numerical[i]], dtype=torch.float32)
        pred1, pred2 = model(cat, num)
        pred1 = torch.sigmoid(pred1).detach().numpy()[0].tolist()
        sub.iloc[i, 1:] = np.clip(pred1, p_min, p_max)
        pass

    sub.iloc[test_features['cp_type'] == 'ctl_vehicle', 1:] = 0
    sub.to_csv(infer_result_dir, index=False)




