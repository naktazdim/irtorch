from dataclasses import dataclass

import numpy as np
import torch


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

    def to_tensors(self) -> "GaussianPriorTensors":
        return GaussianPriorTensors(
            torch.from_numpy(self.mean).float(),
            torch.from_numpy(self.std).float()
        )


@dataclass(frozen=True)
class GaussianPriorTensors(object):
    mean: torch.Tensor
    std: torch.Tensor

    def __post_init__(self):
        assert self.mean.shape == self.std.shape

    def log_normal(self, x: torch.Tensor) -> torch.Tensor:
        # 正規化項 (1/√(2π)σ) は推定に影響しないので省略してある
        return -(torch.sum(((x - self.mean) / self.std) ** 2)) / 2.0
