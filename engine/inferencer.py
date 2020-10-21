# author: yx
# date: 2020/10/19 9:20

import logging
import torch
import os
import csv
from tqdm import trange
from data.datasets.LishMoa import data_preprocess

import numpy as np

def do_inference(
        cfg,
        model
):
    model.eval()

    infer_result_dir = os.path.join(cfg.INFERENCE_RESULT_DIR, 'submission.csv')
    submission_file = cfg.DATASET.SUBMISSION
    categories, numerical = data_preprocess(cfg, indices=None, is_train=False)

    f1 = open(submission_file, 'r', encoding='utf-8')
    f2 = open(infer_result_dir, 'w', encoding='utf-8', newline='')


    contents = f1.readlines()

    csv_writer = csv.writer(f2)

    csv_writer.writerow(contents.pop(0).split(','))

    for i in trange(len(categories)):
        cat = torch.tensor([categories[i]])
        num = torch.tensor([numerical[i]], dtype=torch.float32)
        pred1, pred2 = model(cat, num)
        pred1 = torch.sigmoid(pred1).detach().numpy()[0].tolist()
        pred1.insert(0, contents[i].split(',')[0])
        csv_writer.writerow(pred1)
        pass

    f1.close()
    f2.close()
    pass



