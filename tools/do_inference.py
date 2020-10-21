# author: yx
# date: 2020/10/19 9:00

import os
import torch
from os import mkdir
from configs import cfg
from modeling import build_model
from engine.inferencer import do_inference

def main(cfg):
    infer_result_dir = cfg.INFERENCE_RESULT_DIR

    model = build_model(cfg)
    state_dict = torch.load(cfg.TEST.WEIGHT)

    try:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict['model'].items():
            name = k
            new_state_dict[name] = v
            pass
        model.load_state_dict(new_state_dict)
        pass
    except Exception as e:
        print(e)

    if infer_result_dir and not os.path.exists(infer_result_dir):
        mkdir(infer_result_dir)

    do_inference(cfg, model)
    pass

if __name__ == '__main__':
    main(cfg)
    pass
