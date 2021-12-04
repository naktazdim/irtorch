from typing import Callable, Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

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
        self.loss_total = 0.0

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters())

    def forward(self, indices):
        return self.model.forward(indices)

    def training_step(self, batch, batch_idx):
        loss = self.forward(*batch)
        self.loss_total += loss
        return {"loss": loss}

    def on_epoch_end(self):
        self.loss_total = 0.0

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def validation_step(self):
        pass  # dummy implementation to enable validation

    def validation_epoch_end(self, _):
        return {
            "log_posterior": -self.loss_total,
            "log": {"log_posterior": -self.loss_total}
        }


class OutputBestEstimates(pl.Callback):
    def __init__(self, estimator: GRMEstimator, callback: Callable[[GRMOutputs], Any]):
        self.estimator = estimator
        self.callback = callback
        self.best = -np.inf

    def on_validation_end(self, trainer: pl.Trainer, _):
        log_posterior = trainer.callback_metrics.get("log_posterior")
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

    pl.Trainer(
        default_root_dir=log_dir,
        callbacks=callbacks,
        checkpoint_callback=False,
        max_epochs=n_iter
    ).fit(estimator)
