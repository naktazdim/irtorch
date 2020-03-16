from dataclasses import dataclass
from typing import Union

import torch
from torch import Tensor


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
