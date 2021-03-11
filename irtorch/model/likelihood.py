import torch
from torch import Tensor
import torch.nn.functional as F


def sum_log1mexp(x: Tensor) -> Tensor:
    """
    要素ごとの log(1 - exp(-x)) を計算し、その和を返す。

    素朴に計算すると桁落ちで誤差が大きくなったりinfinityが出たりする。
    see: https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf

    :param x: x
    :return: log(1 - exp(-x)) の総和
    """
    cutoff = 0.69314718056
    greater_index = (x.data > cutoff)
    greater = x[greater_index]
    lesser = x[~greater_index]
    greater_log1mexp = torch.log1p(-torch.exp(-greater))
    lesser_log1mexp = torch.log(-torch.expm1(-lesser))
    return torch.sum(greater_log1mexp) + torch.sum(lesser_log1mexp)


def log_likelihood(a: Tensor,
                   b_lower: Tensor,
                   b_upper: Tensor,
                   t: Tensor) -> Tensor:
    """
    Graded Response Model の log likelihood を計算する。

    Σ log( sigmoid(a * (t - b_lower)) - sigmoid(a * (t - b_upper)) )
    だが、素朴に計算すると桁落ちで誤差が大きくなったりするので対策をしている。

    :param a: (n_responses,)
    :param b_lower: (n_responses,)
    :param b_upper: (n_responses,)
    :param t: (n_responses,)
    :return: log likelihood
    """
    return sum_log1mexp(a * (b_upper - b_lower)) + torch.sum(
        -a * (t - b_upper)
        - F.softplus(-a * (t - b_upper))
        - F.softplus(-a * (t - b_lower))
    )
