import pytest

import torch
from torch import Tensor
from irtorch.estimate.model.likelihood import sum_log1mexp, log_likelihood


def naive_sum_log1mexp(x: Tensor) -> Tensor:
    return torch.sum(torch.log(-torch.exp(-x) + 1.0))


def naive_log_likelihood(a: Tensor,
                         b_lower: Tensor,
                         b_upper: Tensor,
                         t: Tensor) -> Tensor:
    return torch.sum(torch.log(torch.sigmoid(a * (t - b_lower)) - torch.sigmoid(a * (t - b_upper))))


@pytest.mark.parametrize("x", [
    (0.01,),
    (0.1,),
    (1.0,),
    (3.0,),
    ([0.01, 0.1, 1.0, 3.0],)
])
def test_sumlog1exp(x):
    x = torch.tensor(x)
    assert sum_log1mexp(x).item() == pytest.approx(naive_sum_log1mexp(x).item())


def test_log_likelihood():
    a = torch.tensor([0.1, 0.2])
    b_lower = torch.tensor([0.3, 0.4])
    b_upper = torch.tensor([0.5, 0.6])
    t = torch.tensor([0.7, 0.8])
    assert log_likelihood(a, b_lower, b_upper, t).item() \
           == pytest.approx(naive_log_likelihood(a, b_lower, b_upper, t).item())
