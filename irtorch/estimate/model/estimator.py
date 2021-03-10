import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl

from irtorch.estimate.model.data import GRMInputs
from irtorch.estimate.model.module import GradedResponseModel


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
