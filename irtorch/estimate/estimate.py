import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from irtorch.estimate.entities import Dataset
from irtorch.estimate.model.data import GRMInputs
from irtorch.estimate.converter import Converter

from irtorch.estimate.model import GradedResponseModel


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
    def __init__(self, dir_path: str, converter: Converter, estimator: GRMEstimator):
        self.dir_path = dir_path
        self.converter = converter
        self.estimator = estimator
        self.best = -np.inf

    def on_validation_end(self, trainer: pl.Trainer, _):
        log_posterior = trainer.callback_metrics.get("log_posterior")
        if log_posterior > self.best:
            self.best = log_posterior
            self.converter.outputs_to_dfs(self.estimator.model.grm_outputs()).to_csvs(self.dir_path)


def estimate(
        response_df: pd.DataFrame,
        out_dir: str,
        log_dir: str,
        n_iter: int,
        batch_size: int,
        patience: int = None,
        level_df: pd.DataFrame = None,
):
    converter = Converter()
    grm_inputs = converter.inputs_from_dfs(Dataset(response_df, level_df))
    estimator = GRMEstimator(grm_inputs, batch_size)
    callbacks = [OutputBestEstimates(out_dir, converter, estimator)]
    if patience:
        callbacks.append(
            EarlyStopping(
                monitor="log_posterior",
                mode="max",
                patience=patience
            )
        )

    trainer = pl.Trainer(
        default_root_dir=log_dir,
        callbacks=callbacks,
        checkpoint_callback=False,
        max_epochs=n_iter
    )
    trainer.fit(estimator)
