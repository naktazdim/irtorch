from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GaussianPrior(object):
    mean: np.ndarray
    std: np.ndarray

    def __post_init__(self):
        assert self.mean.shape == self.std.shape

    @property
    def shape(self):
        return self.mean.shape

    def reshape(self, *args: int) -> "GaussianPrior":
        return GaussianPrior(self.mean.reshape(*args),
                             self.std.reshape(*args))
