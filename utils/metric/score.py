# author: yx
# date: 2020/10/16 19:40

from __future__ import division

from layers.log_loss import log_loss_multi
from ignite.exceptions import NotComputableError
from ignite.metrics import Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced


class CV_Score(Metric):
    _required_output_keys = None

    def __init__(self, output_transform=lambda x: x,
                 batch_size=lambda x: len(x), device=None):
        super(CV_Score, self).__init__(output_transform, device=device)
        self._batch_size = batch_size

    @reinit__is_reduced
    def reset(self):
        self._sum = 0
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output):
        y_pred1, y_pred2, y1, y2 = output
        average_score = log_loss_multi(y1, y_pred1)

        if len(average_score.shape) != 0:
            raise ValueError('log_loss_multi did not return the average_score.')

        N = self._batch_size(y1)
        self._sum += average_score.item() * N
        self._num_examples += N

    @sync_all_reduce("_sum", "_num_examples")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError(
                'Score must have at least one example before it can be computed.')
        return self._sum / self._num_examples