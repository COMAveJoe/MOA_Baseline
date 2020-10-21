# author: yx
# date: 2020/10/16 15:00

import torch as t
from torch import nn
from torch.nn import functional as F


class MOA_MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        num_cats = cfg.MODEL.NUM_CATS
        cats_emb_size = cfg.MODEL.CATS_EMB_SIZE
        num_numericals = cfg.MODEL.NUM_NUMERICALS
        hidden_size_numericals = cfg.MODEL.HIDDEN_SIZE
        num_class = cfg.MODEL.NUM_CLASS
        aux = cfg.MODEL.AUX

        self.cat_emb1 = nn.Embedding(num_cats[0], cats_emb_size[0], padding_idx=0)
        self.cat_emb2 = nn.Embedding(num_cats[1], cats_emb_size[1], padding_idx=0)
        # self.cat_emb3 = nn.Embedding(num_cats[2], cats_emb_size[2], padding_idx=0)

        self.norms = nn.BatchNorm1d(sum(cats_emb_size) + num_numericals)
        self.dropout = nn.Dropout(0.2)

        self.proj = nn.utils.weight_norm(nn.Linear(sum(cats_emb_size) + num_numericals, hidden_size_numericals))
        self.norm_proj = nn.BatchNorm1d(hidden_size_numericals)
        self.dropout2 = nn.Dropout(0.5)

        hd_1 = hidden_size_numericals // 2
        hd_2 = hd_1 // 2
        self.extractor = nn.Sequential(nn.utils.weight_norm(nn.Linear(hidden_size_numericals, hd_1)),
                                       nn.PReLU(),
                                       nn.BatchNorm1d(hd_1),
                                       nn.Dropout(0.5),
                                       # nn.utils.weight_norm(nn.Linear(hd_1, hd_2)),
                                       # nn.PReLU(),
                                       # nn.BatchNorm1d(hd_2),
                                       # nn.Dropout(0.5)
                                       )
        self.cls = nn.utils.weight_norm(nn.Linear(hd_1, num_class))
        self.cls_aux = None
        if aux is not None:
            self.cls_aux = nn.utils.weight_norm(nn.Linear(hd_1, aux))

    def forward(self, x_cat, x_num):
        cat_features = t.cat([self.cat_emb1(x_cat[:, 0]), self.cat_emb2(x_cat[:, 1])], dim=1)
        all_features = t.cat([cat_features, x_num], dim=1)
        all_features = self.norms(all_features)
        all_features = self.dropout(all_features)

        proj_features = self.proj(all_features)
        proj_features = self.norm_proj(F.relu(proj_features))
        proj_features = self.dropout2(proj_features)

        features_reduced = self.extractor(proj_features)

        outputs = self.cls(features_reduced)
        if self.cls_aux is not None:
            outputs2 = self.cls_aux(features_reduced)
            return outputs, outputs2
        return outputs


class MOA_MLPv2(nn.Module):
    def __init__(self, num_cats=[2, 3, 2], cats_emb_size=[2, 2, 2], num_numericals=872, hidden_size_numericals=2048,
                 num_class=206, aux=None):
        super().__init__()
        self.cat_emb1 = nn.Embedding(num_cats[0], cats_emb_size[0], padding_idx=0)
        self.cat_emb2 = nn.Embedding(num_cats[1], cats_emb_size[1], padding_idx=0)
        self.cat_emb3 = nn.Embedding(num_cats[2], cats_emb_size[2], padding_idx=0)

        self.projection_numericals = nn.Linear(num_numericals, hidden_size_numericals)
        self.norm_numericals = nn.BatchNorm1d(hidden_size_numericals)
        self.dropout = nn.Dropout(0.5)

        self.proj = nn.Linear(sum(cats_emb_size) + hidden_size_numericals, 2048)
        self.norm_proj = nn.BatchNorm1d(2048)

        hd_1 = hidden_size_numericals // 2
        hd_2 = hd_1 // 2
        self.extractor = nn.Sequential(nn.Linear(2048, hd_1),
                                       nn.ReLU(),
                                       nn.BatchNorm1d(hd_1),
                                       nn.Dropout(0.25),
                                       nn.Linear(hd_1, hd_2),
                                       nn.ReLU(),
                                       nn.BatchNorm1d(hd_2),
                                       nn.Dropout(0.25))

        self.cls = nn.Linear(hd_2, num_class)
        self.cls_aux = None
        if aux is not None:
            self.cls_aux = nn.Linear(hd_2, aux)

    def forward(self, x_cat, x_num):
        cat_features = t.cat([self.cat_emb1(x_cat[:, 0]), self.cat_emb2(x_cat[:, 1]), self.cat_emb3(x_cat[:, 2])],
                                 dim=1)

        num_features = self.projection_numericals(x_num)
        num_features = self.norm_numericals(F.relu(num_features))

        all_features = t.cat([cat_features, num_features], dim=1)
        all_features = self.dropout(all_features)

        all_features = F.relu(self.proj(all_features))
        all_features = self.norm_proj(all_features)

        features_reduced = self.extractor(all_features)

        outputs = self.cls(features_reduced)
        if self.cls_aux is not None:
            outputs2 = self.cls_aux(features_reduced)
            return outputs, outputs2
        return outputs