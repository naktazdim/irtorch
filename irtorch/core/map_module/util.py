import torch
import torch.nn as nn
import torch.nn.functional as F


def parameter(*size: int) -> nn.Parameter:
    return nn.Parameter(torch.zeros(*size), requires_grad=True)


def positive(tensor: torch.Tensor) -> torch.Tensor:
    return F.softplus(tensor)  # ReLU + eps とかだとうまくいかない (おそらく勾配消失のせい)
