import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import fire

from irtorch.dataset import Dataset, Converter
from irtorch.model import GRMEstimator


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
        dataset: Dataset,
        out_dir: str,
        log_dir: str,
        n_iter: int,
        batch_size: int,
        patience: int = None,
):
    converter = Converter()
    grm_inputs = converter.inputs_from_dfs(dataset)
    estimator = GRMEstimator(grm_inputs, batch_size)
    callbacks = [OutputBestEstimates(out_dir, converter, estimator)]
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


def estimate_cli(
        response: str,
        level: str = None,
        n_iter: int = 1000,
        batch_size: int = 1000,
        patience: int = None,
        out_dir: str = ".",
        log_dir: str = "."
):
    estimate(
        Dataset.from_csvs(response, level),
        n_iter=n_iter,
        batch_size=batch_size,
        patience=patience,
        out_dir=out_dir,
        log_dir=log_dir,
    )


if __name__ == "__main__":
    fire.Fire(estimate_cli)
