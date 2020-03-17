import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
import numpy as np

from .likelihood import log_likelihood2
from .prior import Normal, InverseGamma


def positive(tensor: Tensor) -> Tensor:
    return F.softplus(tensor)  # ReLU + eps とかだとうまくいかない (おそらく勾配消失のせい)


class GRMMAPModule(nn.Module):
    """
    Graded Response Model のパラメータを MAP 推定するための Module
    """

    def __init__(self,
                 response_array: np.ndarray,
                 level_index: np.ndarray = None):
        super().__init__()

        n_items = response_array[:, 0].max() + 1
        n_persons = response_array[:, 1].max() + 1
        n_grades = response_array[:, 2].max()
        self.n_responses = len(response_array)
        self.dataset = TensorDataset(torch.tensor(response_array).long())

        # パラメータ
        def parameter(*size: int) -> nn.Parameter:
            return nn.Parameter(torch.zeros(*size), requires_grad=True)
        self.a_ = parameter(n_items)
        self.b_base_ = parameter(n_items, 1)
        self.b_diff_ = parameter(n_items, n_grades - 2)
        self.t = parameter(n_persons)

        # その他 (b_prior の定義は下に)
        self.a_prior = Normal()
        self.t_prior = Normal()

        # 階層ベイズ用
        if level_index is None:
            self.is_hierarchical = False
        else:
            self.is_hierarchical = True
            n_levels = level_index.max() + 1
            self.b_prior_mean = parameter(n_levels, n_grades - 1)
            self.b_prior_std_ = parameter(n_levels, n_grades - 1)
            self.b_prior_mean_prior = Normal()
            self.b_prior_std_prior = InverseGamma()
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
        if self.is_hierarchical:
            return Normal(self.b_prior_mean[self.level_index, :],
                          self.b_prior_std[self.level_index, :])
        else:
            return Normal(0.0, 1.0)

    def forward(self, indices: Tensor) -> Tensor:
        """

        :param indices: item, person, response [shape=(n_responses, 3)]
        :return:
        """
        log_prior = self.a_prior.log_pdf(self.a) + self.b_prior.log_pdf(self.b) + self.t_prior.log_pdf(self.t)
        log_likelihood = log_likelihood2(self.a, self.b, self.t, indices[:, 0], indices[:, 1], indices[:, 2])

        if self.is_hierarchical:
            log_prior += self.b_prior_mean_prior.log_pdf(self.b_prior_mean) + \
                         self.b_prior_std_prior.log_pdf(self.b_prior_std)

        # SGDでデータの一部を渡すことを想定してpriorに補正をかけている
        log_posterior = log_likelihood + log_prior * (indices.shape[0] / self.n_responses)

        return -log_posterior  # 「事後確率の対数のマイナスを最小化」⇔「事後確率を最大化」
