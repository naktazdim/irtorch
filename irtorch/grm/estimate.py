import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl

from irtorch.grm.converter import GRMDataConverter
from irtorch.grm.map_module import GRMMAPModule


class GRMEstimator(pl.LightningModule):
    def __init__(self,
                 response_df: pd.DataFrame,
                 a_prior_df: pd.DataFrame = None,
                 b_prior_df: pd.DataFrame = None,
                 t_prior_df: pd.DataFrame = None):
        super(GRMEstimator, self).__init__()

        self.converter = GRMDataConverter(response_df)

        self.model = GRMMAPModule(
            a_prior=self.converter.make_a_prior(a_prior_df),
            b_prior=self.converter.make_b_prior(b_prior_df),
            t_prior=self.converter.make_t_prior(t_prior_df),
            num_responses_total=len(response_df)
        )

        indices = np.c_[self.converter.make_item_array(),
                        self.converter.make_person_array(),
                        self.converter.make_response_array()]
        self.dataset = TensorDataset(torch.tensor(indices).long())

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
        return DataLoader(self.dataset, batch_size=1000, shuffle=True)

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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("response", type=str)
    p.add_argument("--a-prior", type=str)
    p.add_argument("--b-prior", type=str)
    p.add_argument("--t-prior", type=str)
    p.add_argument("-o", "--out-dir", type=str)
    args = p.parse_args()

    response_df = pd.read_csv(args.response)
    estimator = GRMEstimator(response_df)

    trainer = pl.Trainer(default_save_path=args.out_dir, max_epochs=1)
    trainer.fit(estimator)
    estimator.output_results(args.out_dir)


if __name__ == "__main__":
    main()
