# author: yx
# date: 2020/10/16 19:40

from __future__ import division

from configs import cfg
from ignite.exceptions import NotComputableError
from ignite.metrics import Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced


class LogLoss(Metric):
    _required_output_keys = None

    def __init__(self, loss_fns, output_transform=lambda x: x,
                 batch_size=lambda x: len(x), device=None):
        super(LogLoss, self).__init__(output_transform, device=device)
        self._loss_fns = loss_fns
        self._batch_size = batch_size

    @reinit__is_reduced
    def reset(self):
        self._sum = 0
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output):
        y_pred1, y_pred2, y1, y2 = output
        loss1 = self._loss_fns[0](y_pred1, y1).mean(0).mean()
        loss2 = self._loss_fns[1](y_pred2, y2).mean(0).mean()
        alpha = cfg.SOLVER.ALPHA
        average_loss = loss1 + alpha * loss2

        if len(average_loss.shape) != 0:
            raise ValueError('loss_fn did not return the average loss.')

        N = self._batch_size(y1)
        self._sum += average_loss.item() * N
        self._num_examples += N

    @sync_all_reduce("_sum", "_num_examples")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError(
                'Loss must have at least one example before it can be computed.')
        return self._sum / self._num_examples


