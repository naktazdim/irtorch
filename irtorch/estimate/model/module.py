import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from irtorch.estimate.model.likelihood import log_likelihood
from irtorch.estimate.model.prior import Normal, InverseGamma
from irtorch.estimate.model.data import GRMShapes, GRMOutputs


def _parameter(*size: int) -> nn.Parameter:
    return nn.Parameter(torch.zeros(*size), requires_grad=True)


def _positive(tensor: torch.Tensor) -> torch.Tensor:
    return F.softplus(tensor)  # ReLU + eps とかだとうまくいかない (おそらく勾配消失のせい)


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.cpu().detach().numpy()


class GradedResponseModel(nn.Module):
    def __init__(self, shapes: GRMShapes,
                 level_index: np.ndarray):
        super().__init__()

        self.n_items = shapes.n_items
        self.n_grades = shapes.n_grades
        self.n_responses = shapes.n_responses
        n_levels = level_index.max() + 1

        self.level_index = torch.from_numpy(level_index).long()

        self.a_ = _parameter(self.n_items)
        self.b_base_ = _parameter(self.n_items, 1)
        self.b_diff_ = _parameter(self.n_items, self.n_grades - 2)
        self.t = _parameter(shapes.n_persons)
        self.b_prior_mean = _parameter(n_levels, self.n_grades - 1)
        self.b_prior_std_ = _parameter(n_levels, self.n_grades - 1)

    @property
    def a(self) -> Tensor:
        return _positive(self.a_)

    @property
    def b(self):
        return torch.cumsum(
            torch.cat([self.b_base_, _positive(self.b_diff_)], dim=1),
            dim=1
        )

    @property
    def b_prior_std(self):
        return _positive(self.b_prior_std_)

    def log_prior(self) -> Tensor:
        a_prior = Normal()
        b_prior = Normal(self.b_prior_mean[self.level_index, :],
                         self.b_prior_std[self.level_index, :])
        t_prior = Normal()
        b_prior_mean_prior = Normal()
        b_prior_std_prior = InverseGamma()

        return a_prior.log_pdf(self.a) + b_prior.log_pdf(self.b) + t_prior.log_pdf(self.t) + \
               b_prior_mean_prior.log_pdf(self.b_prior_mean) + \
               b_prior_std_prior.log_pdf(self.b_prior_std)

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

    def grm_outputs(self) -> GRMOutputs:
        return GRMOutputs(
            _to_numpy(self.a),
            _to_numpy(self.b),
            _to_numpy(self.t),
            _to_numpy(self.b_prior_mean),
            _to_numpy(self.b_prior_std)
        )
