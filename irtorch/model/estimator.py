from typing import Callable, Any
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from irtorch.model.data import GRMInputs, GRMOutputs
from irtorch.model.module import GradedResponseModel


class GRMEstimator(pl.LightningModule):
    def __init__(self,
                 inputs: GRMInputs,
                 batch_size: int,
                 ):
        super(GRMEstimator, self).__init__()

        self.model = GradedResponseModel(inputs.shapes, inputs.level_array)
        self.batch_size = batch_size
        self.dataset = TensorDataset(torch.tensor(inputs.response_array).long())

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters())

    def forward(self, indices):
        return self.model.forward(indices)

    def training_step(self, batch, batch_idx):
        return self.forward(*batch)

    def training_epoch_end(self, outputs):
        self.log("log_posterior",
                 -sum([output["loss"] for output in outputs]))

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)


class OutputBestEstimates(pl.Callback):
    def __init__(self, estimator: GRMEstimator, callback: Callable[[GRMOutputs], Any]):
        self.estimator = estimator
        self.callback = callback
        self.best = -np.inf

    def on_train_epoch_end(self, trainer: pl.Trainer, *args):
        log_posterior = trainer.callback_metrics["log_posterior"]
        if log_posterior > self.best:
            self.best = log_posterior
            self.callback(self.estimator.model.grm_outputs())


def estimate(
        grm_inputs: GRMInputs,
        log_dir: str,
        n_iter: int,
        batch_size: int,
        callback: Callable[[GRMOutputs], Any],
        patience: int = None,
):
    estimator = GRMEstimator(grm_inputs, batch_size)
    callbacks = [OutputBestEstimates(estimator, callback)]
    if patience:
        callbacks.append(EarlyStopping(monitor="log_posterior",
                                       mode="max",
                                       patience=patience))

    trainer = pl.Trainer(
        logger=TensorBoardLogger(log_dir, name="lightning_logs", default_hp_metric=False),
        callbacks=callbacks,
        enable_checkpointing=False,
        max_epochs=n_iter
    )
    with warnings.catch_warnings():
        # 「DataLoaderのnum_workersが少ない」というUserWarningを無視 (ただのTensorDatasetなので並列化する意味がない)
        warnings.filterwarnings("ignore",
                                category=UserWarning,
                                message=".*num_workers",
                                module="pytorch_lightning.trainer.data_loading")
        trainer.fit(estimator)
