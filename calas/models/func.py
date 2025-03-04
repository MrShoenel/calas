"""
Functional module with often-required static functionality.
"""

import math
import torch
from torch import Tensor



def normal_cdf(x: Tensor, loc: Tensor, scale: Tensor) -> Tensor:
    return 0.5 * (1 + torch.erf((x - loc) * scale.reciprocal() / math.sqrt(2)))


def std_normal_cdf(x: Tensor) -> Tensor:
    return normal_cdf(x=x, loc=torch.zeros_like(input=x), scale=torch.ones_like(input=x))


def normal_ppf(q: Tensor, loc: Tensor, scale: Tensor) -> Tensor:
    return loc + scale * torch.erfinv(2 * q - 1) * math.sqrt(2)


def std_normal_ppf(q: Tensor) -> Tensor:
    return normal_ppf(q=q, loc=torch.zeros_like(input=q), scale=torch.ones_like(input=q))


def normal_ppf_safe(q: Tensor, loc: Tensor, scale: Tensor, tol: float=1e-7) -> Tensor:
    if q.dtype.itemsize < 4:
        tol *= 1e4
    q = torch.clip(input=q, min=tol, max=1.-tol)
    return normal_ppf(q=q, loc=loc, scale=scale)


def std_normal_ppf_safe(q: Tensor, tol: float=1e-7) -> Tensor:
    return normal_ppf_safe(q=q, tol=tol, loc=torch.zeros_like(input=q), scale=torch.ones_like(input=q))
