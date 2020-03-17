import torch
from torch import Tensor
import numpy as np

from irtorch.core.prior import Normal, InverseGamma
from irtorch.core.module.base import GradedResponseModel
from irtorch.core.util import positive, parameter


class HierarchicalGradedResponseModel(GradedResponseModel):
    def __init__(self,
                 response_array: np.ndarray,
                 level_index: np.ndarray):
        super().__init__(response_array)

        n_levels = level_index.max() + 1

        self.b_prior_mean = parameter(n_levels, self.n_grades - 1)
        self.b_prior_std_ = parameter(n_levels, self.n_grades - 1)

        self.b_prior_mean_prior = Normal()
        self.b_prior_std_prior = InverseGamma()
        self.level_index = torch.from_numpy(level_index).long()

    @property
    def b_prior_std(self):
        return positive(self.b_prior_std_)

    def log_prior(self) -> Tensor:
        self.b_prior = Normal(self.b_prior_mean[self.level_index, :],
                              self.b_prior_std[self.level_index, :])
        return super().log_prior() + \
               self.b_prior_mean_prior.log_pdf(self.b_prior_mean) + \
               self.b_prior_std_prior.log_pdf(self.b_prior_std)
