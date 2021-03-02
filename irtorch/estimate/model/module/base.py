import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import TensorDataset
import numpy as np

from irtorch.estimate.model.likelihood import log_likelihood
from irtorch.estimate.model.prior import Normal
from irtorch.estimate.model.util import positive, parameter


class GradedResponseModel(nn.Module):
    def __init__(self, response_array: np.ndarray):
        super().__init__()

        self.n_items = response_array[:, 0].max() + 1
        n_persons = response_array[:, 1].max() + 1
        self.n_grades = response_array[:, 2].max()
        self.n_responses = len(response_array)
        self.dataset = TensorDataset(torch.tensor(response_array).long())

        self.a_ = parameter(self.n_items)
        self.b_base_ = parameter(self.n_items, 1)
        self.b_diff_ = parameter(self.n_items, self.n_grades - 2)
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
        item_index = indices[:, 0]
        person_index = indices[:, 1]
        response_index = indices[:, 2]

        inf = 1.0e3  # 本当は np.inf を入れたいが、それをすると微分のときに NaN が発生するようなので十分大きな値で代用
        infs = torch.full((self.n_items, 1), inf)
        b_ = torch.cat((-infs, self.b, infs), dim=1)
        b_lower = b_[item_index, response_index - 1]
        b_upper = b_[item_index, response_index]
        
        return log_likelihood(self.a[item_index], b_lower, b_upper, self.t[person_index])

    def log_posterior(self, indices: Tensor) -> Tensor:
        # SGDでデータの一部を渡すことを想定してpriorに補正をかけている
        return self.log_likelihood(indices) + self.log_prior() * (indices.shape[0] / self.n_responses)

    def forward(self, indices: Tensor) -> Tensor:
        """

        :param indices: item, person, response [shape=(n_responses, 3)]
        :return:
        """
        return -self.log_posterior(indices)  # 「事後確率の対数のマイナスを最小化」⇔「事後確率を最大化」
