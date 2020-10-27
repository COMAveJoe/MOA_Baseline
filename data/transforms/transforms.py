# author: yx
# date: 2020/10/26 11:40
from configs import cfg
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from layers.gauss_rank_scaler import GaussRankScaler

def scale_minmax(col):
    return (col - col.min()) / (col.max() - col.min())

def scale_norm(col):
    return (col - col.mean()) / col.std()

def transform(scale, features):
    cols_numeric = [feat for feat in list(features.columns) if feat not in ["sig_id", "cp_type", "cp_time", "cp_dose"]]

    if scale == "boxcox":
        features[cols_numeric] = features[cols_numeric].apply(scale_minmax, axis=0)
        trans = []
        for feat in cols_numeric:
            trans_var, lambda_var = stats.boxcox(features[feat].dropna() + 1)
            trans.append(scale_minmax(trans_var))
        features[cols_numeric] = np.asarray(trans).T

    elif scale == "norm":
        features[cols_numeric] = features[cols_numeric].apply(scale_norm, axis=0)

    elif scale == "minmax":
        features[cols_numeric] = features[cols_numeric].apply(scale_minmax, axis=0)

    elif scale == "rankgauss":
        ### Rank Gauss ###
        scaler = GaussRankScaler()

        features[cols_numeric] = scaler.fit_transform(features[cols_numeric])
    else:
        pass

    numerical = features[cols_numeric].values
    return numerical

def pca(features):
    GENES = [col for col in features.columns if col.startswith("g-")]
    CELLS = [col for col in features.columns if col.startswith("c-")]

    pca_genes = PCA(n_components=80,
                    random_state=42).fit_transform(features[GENES])
    pca_cells = PCA(n_components=10,
                    random_state=42).fit_transform(features[CELLS])
    
    pca_genes = pd.DataFrame(pca_genes, columns=[f"pca_g-{i}" for i in range(80)])
    pca_cells = pd.DataFrame(pca_cells, columns=[f"pca_c-{i}" for i in range(10)])

    features = pd.concat([features, pca_genes, pca_cells], axis=1)