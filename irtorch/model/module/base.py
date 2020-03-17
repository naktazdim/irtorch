import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import TensorDataset
import numpy as np

from irtorch.model.likelihood import log_likelihood2
from irtorch.model.prior import Normal
from irtorch.model.util import positive, parameter


class GradedResponseModel(nn.Module):
    def __init__(self, response_array: np.ndarray):
        super().__init__()

        n_items = response_array[:, 0].max() + 1
        n_persons = response_array[:, 1].max() + 1
        self.n_grades = response_array[:, 2].max()
        self.n_responses = len(response_array)
        self.dataset = TensorDataset(torch.tensor(response_array).long())

        self.a_ = parameter(n_items)
        self.b_base_ = parameter(n_items, 1)
        self.b_diff_ = parameter(n_items, self.n_grades - 2)
        self.t = parameter(n_persons)

        self.a_prior = Normal()
        self.b_prior = Normal()
        self.t_prior = Normal()

    @property
    def a(self) -> Tensor:
        return positive(self.a_)

    @property
    def b(self):
        return torch.cumsum(
            torch.cat([self.b_base_, positive(self.b_diff_)], dim=1),
            dim=1
        )

    def log_prior(self) -> Tensor:
        return self.a_prior.log_pdf(self.a) + self.b_prior.log_pdf(self.b) + self.t_prior.log_pdf(self.t)

    def log_likelihood(self, indices: Tensor) -> Tensor:
        return log_likelihood2(self.a, self.b, self.t, indices[:, 0], indices[:, 1], indices[:, 2])

    def log_posterior(self, indices: Tensor) -> Tensor:
        # SGDでデータの一部を渡すことを想定してpriorに補正をかけている
        return self.log_likelihood(indices) + self.log_prior() * (indices.shape[0] / self.n_responses)

    def forward(self, indices: Tensor) -> Tensor:
        """

        :param indices: item, person, response [shape=(n_responses, 3)]
        :return:
        """
        return -self.log_posterior(indices)  # 「事後確率の対数のマイナスを最小化」⇔「事後確率を最大化」
