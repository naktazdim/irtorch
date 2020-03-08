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
    log_likelihood2() との違いは入力の形式。

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


def log_likelihood2(a: Tensor,
                    b: Tensor,
                    t: Tensor,
                    item_index: Tensor,
                    person_index: Tensor,
                    response_index: Tensor) -> Tensor:
    """
    Graded Response Model の log likelihood を計算する。

    log_likelihood() との違いは入力の形式。

    :param a: (n_items,)
    :param b: (n_items, n_grades - 1)
    :param t: (n_persons,)
    :param item_index: (n_responses,)
    :param person_index: (n_responses,)
    :param response_index: (n_responses,)
    :return: log likelihood
    """
    # response が端の値 (1 または n_grades) のときに場合分けをしなくてよいように b に sentinel (-inf と inf) を置いている
    inf = 1.0e3  # 本当は np.inf を入れたいが、それをすると微分のときに NaN が発生するようなので十分大きな値で代用
    infs = torch.full((b.shape[0], 1), inf)
    b_ = torch.cat((-infs, b, infs), dim=1)

    return log_likelihood(a[item_index],
                          b_[item_index, response_index - 1],
                          b_[item_index, response_index],
                          t[person_index])
