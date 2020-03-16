from dataclasses import dataclass
from typing import Union

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .likelihood import log_likelihood2


def positive(tensor: Tensor) -> Tensor:
    return F.softplus(tensor)  # ReLU + eps とかだとうまくいかない (おそらく勾配消失のせい)


@dataclass()
class Normal:
    mean: Union[float, Tensor] = 0.0
    std: Union[float, Tensor] = 1.0

    def __post_init__(self):
        # std が float だったら scalar tensor に直しておく (下の torch.log(self.std) のため)
        self.std = torch.tensor(self.std) if isinstance(self.std, float) else self.std

    def log_pdf(self, x: Tensor) -> Tensor:
        # 正規化項 (1/√(2π)) は推定に影響しないので省略してある
        return torch.sum(-(((x - self.mean) / self.std) ** 2) / 2.0 - torch.log(self.std))


@dataclass()
class InverseGamma:
    alpha: float = 1.0
    beta: float = 1.0

    def log_pdf(self, x: Tensor) -> Tensor:
        # 正規化項 (beta^alpha / gamma(alpha)) は推定に影響しないので省略してある
        return torch.sum(-(self.alpha + 1.0) * torch.log(x) - torch.tensor(self.beta) / x)


class GRMMAPModule(nn.Module):
    """
    Graded Response Model のパラメータを MAP 推定するための Module
    """

    def __init__(self,
                 n_items: int,
                 n_persons: int,
                 n_grades: int,
                 n_responses: int,
                 n_labels: int,
                 level_index: np.ndarray):
        super().__init__()

        # パラメータ
        def parameter(*size: int) -> nn.Parameter:
            return nn.Parameter(torch.zeros(*size), requires_grad=True)
        self.a_ = parameter(n_items)
        self.b_base_ = parameter(n_items, 1)
        self.b_diff_ = parameter(n_items, n_grades - 2)
        self.t = parameter(n_persons)
        self.b_prior_mean = parameter(n_labels, n_grades - 1)
        self.b_prior_std_ = parameter(n_labels, n_grades - 1)

        # その他
        self.a_prior = Normal()
        self.t_prior = Normal()
        self.b_prior_mean_prior = Normal()
        self.b_prior_std_prior = InverseGamma()
        self.n_responses = n_responses
        self.level_index = torch.from_numpy(level_index).long()

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
    def b_prior_std(self):
        return positive(self.b_prior_std_)

    @property
    def b_prior(self):
        return Normal(self.b_prior_mean[self.level_index, :],
                      self.b_prior_std[self.level_index, :])

    def forward(self, indices: Tensor) -> Tensor:
        """

        :param indices: item, person, response [shape=(n_responses, 3)]
        :return:
        """
        log_hyperprior = self.b_prior_mean_prior.log_pdf(self.b_prior_mean) + self.b_prior_std_prior.log_pdf(self.b_prior_std)
        log_prior = self.a_prior.log_pdf(self.a) + self.b_prior.log_pdf(self.b) + self.t_prior.log_pdf(self.t)
        log_likelihood = log_likelihood2(self.a, self.b, self.t, indices[:, 0], indices[:, 1], indices[:, 2])

        # SGDでデータの一部を渡すことを想定してpriorに補正をかけている
        log_posterior = log_likelihood + (log_prior + log_hyperprior) * (indices.shape[0] / self.n_responses)

        return -log_posterior  # 「事後確率の対数のマイナスを最小化」⇔「事後確率を最大化」
