from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from irtorch.converter import GRMDataConverter
from irtorch.map_module import GRMMAPModule


class GRMEstimator(pl.LightningModule):
    def __init__(self,
                 response_df: pd.DataFrame,
                 batch_size: int,
                 level_df: pd.DataFrame = None,
                 ):
        super(GRMEstimator, self).__init__()

        self.converter = GRMDataConverter(response_df, level_df)

        self.model = GRMMAPModule(
            response_array=self.converter.make_response_array(),
            level_index=self.converter.make_level_array()
        )

        self.batch_size = batch_size

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
        return DataLoader(self.model.dataset, batch_size=self.batch_size, shuffle=True)

    def validation_step(self):
        pass  # dummy implementation to enable validation

    def validation_epoch_end(self, _):
        return {
            "log_posterior": -self.loss_total,
            "log": {"log_posterior": -self.loss_total}
        }

    def output_results(self, dir_path: str):
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        self.converter.make_a_df(self.model.a.detach().numpy()).to_csv(dir_path / "a.csv", index=False)
        self.converter.make_b_df(self.model.b.detach().numpy()).to_csv(dir_path / "b.csv", index=False)
        self.converter.make_t_df(self.model.t.detach().numpy()).to_csv(dir_path / "t.csv", index=False)
        if self.converter.is_hierarchical:
            self.converter.make_level_df(
                self.model.b_prior_mean.detach().numpy(),
                self.model.b_prior_std.detach().numpy()
            ).to_csv(dir_path / "b_prior.csv", index=False)


class OutputEstimates(pl.Callback):
    def __init__(self, dir_path: str, estimator: GRMEstimator):
        self.dir_path = dir_path
        self.estimator = estimator
        self.best = -np.inf

    def on_validation_end(self, trainer: pl.Trainer, _):
        log_posterior = trainer.callback_metrics.get("log_posterior")
        if log_posterior > self.best:
            self.best = log_posterior
            self.estimator.output_results(self.dir_path)


def estimate(
        response_df: pd.DataFrame,
        out_dir: str,
        log_dir: str,
        n_iter: int,
        batch_size: int,
        patience: int = None,
        level_df: pd.DataFrame = None,
):
    estimator = GRMEstimator(response_df, batch_size, level_df)
    output_estimates = OutputEstimates(out_dir, estimator)
    early_stop_callback = None if patience is None else EarlyStopping(
        monitor="log_posterior",
        mode="max",
        patience=patience
    )

    trainer = pl.Trainer(
        default_save_path=log_dir,
        early_stop_callback=early_stop_callback,
        callbacks=[output_estimates],
        checkpoint_callback=False,
        max_epochs=n_iter
    )
    trainer.fit(estimator)
