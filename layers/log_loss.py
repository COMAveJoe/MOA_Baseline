import numpy as np
import torch

def log_loss_score(actual, predicted, eps=1e-15):
    """
    :param predicted:   The predicted probabilities as floats between 0-1
    :param actual:      The binary labels. Either 0 or 1.
    :param eps:         Log(0) is equal to infinity, so we need to offset our predicted values slightly by eps from 0 or 1
    :return:            The logarithmic loss between between the predicted probability assigned to the possible outcomes for item i, and the actual outcome.
    """

    p1 = actual * np.log(predicted + eps)
    p0 = (1 - actual) * np.log(1 - predicted + eps)
    loss = p0 + p1

    return -loss.mean()


def log_loss_multi(y_true, y_pred):
    M = y_true.shape[1]
    results = np.zeros(M)

    y_true = y_true.detach().numpy()
    y_pred = torch.sigmoid(y_pred).detach().numpy()

    for i in range(M):
        results[i] = log_loss_score(y_true[:, i], y_pred[:, i])
    return results.mean()
