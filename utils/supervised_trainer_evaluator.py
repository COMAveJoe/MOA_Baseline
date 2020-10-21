# author: yx
# date: 2020/10/16 19:35

import torch
from configs import cfg
from ignite.engine.engine import Engine, State, Events
from ignite.utils import convert_tensor


def _prepare_batch(batch, device=None, non_blocking=False):
    """
    Prepare batch for training: pass to a device with options.
    """
    x1, x2, y1, y2 = batch
    return (convert_tensor(x1, device=device, non_blocking=non_blocking),
            convert_tensor(x2, device=device, non_blocking=non_blocking),
            convert_tensor(y1, device=device, non_blocking=non_blocking),
            convert_tensor(y2, device=device, non_blocking=non_blocking))


def create_supervised_trainer(model, optimizer, loss_fns,
                              device=None, non_blocking=False,
                              prepare_batch=_prepare_batch,
                              output_transform=lambda x1, x2, y1, y2, loss: loss.item()):
    """
    Factory function for creating a trainer for supervised models.

    Args:
        model (`torch.nn.Module`): the model to train.
        optimizer (`torch.optim.Optimizer`): the optimizer to use.
        loss_fns (torch.nn loss functions): the loss functions to use.
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
        non_blocking (bool, optional): if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch (callable, optional): function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.
        output_transform (callable, optional): function that receives 'x1', 'x2','y1', 'y2', 'loss'
            and returns value to be assigned to engine's state.output after each iteration. Default is returning
            `loss.item()`.

    Note: `engine.state.output` for this engine is define by `output_transform` parameter and is the loss
        of the processed batch by default.

    Returns:
        Engine: a trainer engine with supervised update function.
    """
    if device:
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        x1, x2, y1, y2 = prepare_batch(batch, device=device, non_blocking=non_blocking)
        y_pred1, y_pred2 = model(x1, x2)
        loss1 = loss_fns[0](y_pred1, y1).mean(0).mean()
        loss2 = loss_fns[1](y_pred2, y2).mean(0).mean()
        alpha = cfg.SOLVER.ALPHA
        loss = loss1 + alpha * loss2

        loss.backward()
        optimizer.step()
        return output_transform(x1, x2, y1, y2, loss)

    return Engine(_update)


def create_supervised_evaluator(model, metrics=None,
                                device=None, non_blocking=False,
                                prepare_batch=_prepare_batch,
                                output_transform=lambda y_pred1, y_pred2, y1, y2: (y_pred1, y_pred2, y1, y2)):
    """
    Factory function for creating an evaluator for supervised models.

    Args:
        model (`torch.nn.Module`): the model to train.
        metrics (dict of str - :class:`~ignite.metrics.Metric`): a map of metric names to Metrics.
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
        non_blocking (bool, optional): if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch (callable, optional): function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.
        output_transform (callable, optional): function that receives 'y_pred1', 'y_pred2', 'y1', 'y2' and returns
            value to be assigned to engine's state.output after each iteration. Default is returning `(y_pred1, y_pred2,
             y1, y2)` which fits output expected by metrics. If you change it you should use `output_transform`
            in metrics.

    Note: `engine.state.output` for this engine is define by `output_transform` parameter and is
        a tuple of `(batch_pred, batch_y)` by default.

    Returns:
        Engine: an evaluator engine with supervised inference function.
    """
    metrics = metrics or {}

    if device:
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            x1, x2, y1, y2 = prepare_batch(batch, device=device, non_blocking=non_blocking)
            y_pred1, y_pred2 = model(x1, x2)
            return output_transform(y_pred1, y_pred2, y1, y2)

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine