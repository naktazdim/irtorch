import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from .prior import GaussianPrior
from .likelihood import log_likelihood2


def positive(tensor: Tensor) -> Tensor:
    return F.softplus(tensor)  # ReLU + eps とかだとうまくいかない (おそらく勾配消失のせい)


class GRMMAPModule(nn.Module):
    """
    Graded Response Model のパラメータを MAP 推定するための Module
    """

    def __init__(self,
                 a_prior: GaussianPrior, b_prior: GaussianPrior, t_prior: GaussianPrior,
                 num_responses_total: int):
        super().__init__()

        # パラメータ、初期値はすべて Prior の mean
        def p(arr: np.ndarray) -> nn.Parameter:
            return nn.Parameter(torch.from_numpy(arr).float())
        self.a_ = p(a_prior.mean)
        self.b_base_ = p(b_prior.mean[:, 0:1])
        self.b_diff_ = p(np.diff(b_prior.mean, axis=1))
        self.t_ = p(t_prior.mean)

        # その他
        self.a_prior = a_prior.to_tensors()
        self.b_prior = b_prior.to_tensors()
        self.t_prior = t_prior.to_tensors()
        self.num_responses_total = num_responses_total

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

        log_prior = self.a_prior.log_normal(a) + self.b_prior.log_normal(b) + self.t_prior.log_normal(t)
        log_likelihood = log_likelihood2(a, b, t, item_index, person_index, response_index)

        # SGDでデータの一部を渡すことを想定してpriorに補正をかけている
        log_posterior = log_likelihood + log_prior * (indices.shape[0] / self.num_responses_total)

        return -log_posterior  # 「事後確率の対数のマイナスを最小化」⇔「事後確率を最大化」
