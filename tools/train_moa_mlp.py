# author: yx
# date: 2020/10/16 16:30
import os
from os import mkdir

from utils.logger import setup_logger
from configs import cfg
from sklearn.model_selection import KFold
from engine.trainer import do_train
from modeling import build_model
from solver.build import make_optimizer
from layers.loss_functions import bce_with_logits_loss
from data.datasets.LishMoa import get_data_rows_num
from data import make_data_loader


def train(cfg):
    model = build_model(cfg)
    data_rows_num = get_data_rows_num(cfg)

    k_fold = KFold(n_splits=10, shuffle=True, random_state=1)
    n_fold = 1
    for train_idx, val_idx in k_fold.split([i for i in range(1, data_rows_num)]):
        optimizer = make_optimizer(cfg, model)
        train_loader = make_data_loader(cfg, train_idx, is_train=True)
        val_loader = make_data_loader(cfg, val_idx, is_train=True)
        loss_functions = [bce_with_logits_loss, bce_with_logits_loss]
        do_train(
            cfg,
            model,
            train_loader,
            val_loader,
            optimizer,
            loss_functions,
            n_fold
        )
        n_fold += 1
        pass


if __name__ == '__main__':
    # 获取输出路径
    output_dir = cfg.OUTPUT_DIR

    # 如果输出路径不存在则创建
    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)
    logger = setup_logger("MOA_MLP", output_dir, 0)
    logger.info("Running with config:\n{}".format(cfg))
    train(cfg)
