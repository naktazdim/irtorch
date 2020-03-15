from typing import Union

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from .likelihood import log_likelihood2


def positive(tensor: Tensor) -> Tensor:
    return F.softplus(tensor)  # ReLU + eps とかだとうまくいかない (おそらく勾配消失のせい)


def log_normal(x: Tensor, mean: Union[Tensor, float], std: Union[Tensor, float]) -> Tensor:
    # 正規化項 (1/√(2π)σ) は推定に影響しないので省略してある
    return -(torch.sum(((x - mean) / std) ** 2)) / 2.0


class GRMMAPModule(nn.Module):
    """
    Graded Response Model のパラメータを MAP 推定するための Module
    """

    def __init__(self,
                 n_items: int,
                 n_persons: int,
                 n_grades: int,
                 n_responses: int):
        super().__init__()

        # パラメータ
        def parameter(*size: int) -> nn.Parameter:
            return nn.Parameter(torch.zeros(*size), requires_grad=True)
        self.a_ = parameter(n_items)
        self.b_base_ = parameter(n_items, 1)
        self.b_diff_ = parameter(n_items, n_grades - 2)
        self.t_ = parameter(n_persons)

        # その他
        self.a_prior_mean = 0.0
        self.a_prior_std = 1.0
        self.b_prior_mean = 0.0
        self.b_prior_std = 1.0
        self.t_prior_mean = 0.0
        self.t_prior_std = 1.0
        self.n_responses = n_responses

    @property
    def a(self) -> Tensor:
        return positive(self.a_)

    @property
    def b(self):
        return torch.cumsum(
            torch.cat([self.b_base_, positive(self.b_diff_)], dim=1),
            dim=1
        )

    @property
    def t(self):
        return self.t_

    def forward(self, indices: Tensor) -> Tensor:
        """

        :param indices: item, person, response [shape=(n_responses, 3)]
        :return:
        """
        a, b, t = self.a, self.b, self.t
        item_index, person_index, response_index = indices[:, 0], indices[:, 1], indices[:, 2]

        log_prior = \
            log_normal(a, self.a_prior_mean, self.a_prior_std) + \
            log_normal(b, self.b_prior_mean, self.b_prior_std) + \
            log_normal(t, self.t_prior_mean, self.t_prior_std)
        log_likelihood = log_likelihood2(a, b, t, item_index, person_index, response_index)

        # SGDでデータの一部を渡すことを想定してpriorに補正をかけている
        log_posterior = log_likelihood + log_prior * (indices.shape[0] / self.n_responses)

        return -log_posterior  # 「事後確率の対数のマイナスを最小化」⇔「事後確率を最大化」
