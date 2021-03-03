import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from irtorch.estimate.converter import inputs_from_df, GRMMeta, to_csvs
from irtorch.estimate.model import GRMInputs, GRMOutputs

from irtorch.estimate.model import GradedResponseModel, HierarchicalGradedResponseModel


def make_model(inputs: GRMInputs) -> GradedResponseModel:
    n_items = inputs.response_array[:, 0].max() + 1
    n_persons = inputs.response_array[:, 1].max() + 1
    n_grades = inputs.response_array[:, 2].max()
    n_responses = len(inputs.response_array)

    args = [n_items, n_persons, n_grades, n_responses]
    if inputs.level_array is None:
        return GradedResponseModel(*args)
    else:
        return HierarchicalGradedResponseModel(*args, inputs.level_array)


def extract_output(model: GradedResponseModel) -> GRMOutputs:
    is_hierarchical = isinstance(model, HierarchicalGradedResponseModel)
    return GRMOutputs(
        model.a.detach().numpy(),
        model.b.detach().numpy(),
        model.t.detach().numpy(),
        model.b_prior_mean.detach().numpy() if is_hierarchical else None,
        model.b_prior_std.detach().numpy() if is_hierarchical else None,
    )


class GRMEstimator(pl.LightningModule):
    def __init__(self,
                 inputs: GRMInputs,
                 batch_size: int,
                 ):
        super(GRMEstimator, self).__init__()

        self.model = make_model(inputs)
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

    def output_results(self, dir_path: str, meta: GRMMeta):
        to_csvs(extract_output(self.model), dir_path, meta)


class OutputBestEstimates(pl.Callback):
    def __init__(self, dir_path: str, estimator: GRMEstimator, meta: GRMMeta):
        self.dir_path = dir_path
        self.estimator = estimator
        self.best = -np.inf
        self.meta = meta

    def on_validation_end(self, trainer: pl.Trainer, _):
        log_posterior = trainer.callback_metrics.get("log_posterior")
        if log_posterior > self.best:
            self.best = log_posterior
            self.estimator.output_results(self.dir_path, self.meta)


def estimate(
        response_df: pd.DataFrame,
        out_dir: str,
        log_dir: str,
        n_iter: int,
        batch_size: int,
        patience: int = None,
        level_df: pd.DataFrame = None,
):
    meta, inputs = inputs_from_df(response_df, level_df)
    estimator = GRMEstimator(inputs, batch_size)
    callbacks = [OutputBestEstimates(out_dir, estimator, meta)]
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
