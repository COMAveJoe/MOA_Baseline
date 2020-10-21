# author: yx
# date: 2020/10/16 16:10
import torch
import logging
import numpy as np
from ignite.engine import Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage
from utils.metric import LogLoss
from utils.metric import CV_Score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.supervised_trainer_evaluator import create_supervised_trainer, create_supervised_evaluator


def do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_fns,
        n_fold=0
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    epochs = cfg.SOLVER.MAX_EPOCHS
    device = cfg.MODEL.DEVICE
    output_dir = cfg.OUTPUT_DIR
    lr_scheduler = ReduceLROnPlateau(optimizer,
                                     mode='min',
                                     factor=0.1,
                                     threshold=1e-3,
                                     patience=3,
                                     min_lr=5e-6,
                                     eps=1e-08,
                                     verbose=True)

    logger = logging.getLogger("MOA_MLP.train")
    logger.info("Start training")

    trainer = create_supervised_trainer(model, optimizer, loss_fns, device=device)
    evaluator = create_supervised_evaluator(model,
                                            metrics={'log_loss': LogLoss(loss_fns), 'cv_score': CV_Score()},
                                            device=device)
    checkpointer = ModelCheckpoint(output_dir, 'moa_mlp_' + str(n_fold),
                                   n_saved=100, require_empty=False)

    timer = Timer(average=True)

    # automatically adding handlers via a special `attach` method of `RunningAverage` handler
    RunningAverage(output_transform=lambda x: x).attach(trainer, 'avg_loss')

    # automatically adding handlers via a special `attach` method of `Checkpointer` handler
    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=checkpoint_period), checkpointer, {'model': model,
                                                                     'optimizer': optimizer})

    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda engine: lr_scheduler.step)

    # automatically adding handlers via a special `attach` method of `Timer` handler
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1

        if iter % log_period == 0:
            logger.info("K:[{}] Epoch[{}] Iteration[{}/{}] LR: {} Log Loss: {:.3f}"
                        .format(n_fold, engine.state.epoch, iter, len(train_loader), optimizer.param_groups[0]['lr'],
                                engine.state.metrics['avg_loss']))

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics

        log_loss = metrics['log_loss']
        cv_score = metrics['cv_score']
        logger.info("Training Results - K:[{}] Epoch: {} LR: {} Log Loss: {:.3f} CV Score: {:.3f}"
                    .format(n_fold, engine.state.epoch, optimizer.param_groups[0]['lr'], log_loss, cv_score))

    if val_loader is not None:
        # adding handlers using `trainer.on` decorator API
        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            evaluator.run(val_loader)
            metrics = evaluator.state.metrics
            log_loss = metrics['log_loss']
            cv_score = metrics['cv_score']
            logger.info("Validation Results - K:[{}] Epoch: {} LR: {} Log Loss: {:.3f} CV Score: {:.3f}"
                        .format(n_fold, engine.state.epoch, optimizer.param_groups[0]['lr'], log_loss, cv_score)
                        )

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('K:[{}] Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(n_fold, engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        timer.reset()

    trainer.run(train_loader, max_epochs=epochs)

