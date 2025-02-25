import torch
from torch import Tensor
from typing import Literal
from numpy.random import RandomState



def two_moons_likelihood(input: Tensor, complementary: bool=False) -> Tensor:
    """
    Returns the relative likelihood (NOT log likelihood).
    This function is normalized (integrates to 1).
    """
    assert input.shape[0] > 0 and input.shape[1] == 2

    a = torch.abs(input=input[:, 0])
    log_likelihood = (
        -0.5 * ((input.norm(p=2, dim=1) - 2.) / 0.2) ** 2
        -0.5 * ((a - 2.) / 0.3) ** 2
        + torch.log(1. + torch.exp(-4. * a / 0.09)))
    
    # The latter is the normalization constant found by integration.
    lik = torch.exp(log_likelihood) / 2.234940148088843
    if complementary:
        mode = two_moons_likelihood(torch.tensor([[2., 0.]], dtype=torch.float, device=input.device))
        lik = mode - lik

    return lik



def pure_two_moons_likelihood(input: Tensor, mode: Literal['normal', 'tight']='normal') -> Tensor:
    """
    Returns the relative likelihood (NOT log likelihood).
    This function is normalized (integrates to 1).
    """
    assert input.shape[0] > 0 and input.shape[1] == 2

    a = torch.abs(input[:, 0])
    log_likelihood = (
        -0.5 * ((input.norm(p=2, dim=1) - 2.) / 0.2) ** 2
        -0.5 * ((a - 2.) / 0.3) ** 2
        + torch.log(1. + torch.exp(-4. * a / 0.09)))
    
    # cut-offs [0.001, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25] ... and
    # normalization-constants: [2.233745734217589, 2.163348081995403, 2.1264553141907974, 2.0887602662011187, 2.011065854174968, 1.927906023052072, 1.8447403474799984]
    
    cutoff = torch.log(torch.tensor(1e-3 if mode == 'normal' else 0.05, dtype=torch.float, device=input.device))
    
    # Let's cut off something here:
    log_likelihood = torch.where(log_likelihood < cutoff, -torch.inf, log_likelihood)

    norm_const = 2.233745734217589 if mode == 'normal' else 2.163348081995403
    
    # The normalizing constant is slightly smaller.
    return torch.exp(log_likelihood) / norm_const



def two_moons_rejection_sampling(nsamples: int, seed: int=0, complementary: bool=False, pure_moons: bool=False, pure_moons_mode: Literal['normal', 'tight']='normal') -> Tensor:
    lik_fn = (lambda input: pure_two_moons_likelihood(input=input, mode=pure_moons_mode)) if pure_moons else two_moons_likelihood
    mode = lik_fn(torch.tensor([[2., 0]], dtype=torch.float))
    rs = RandomState(seed=seed)
    samples: list[Tensor] = []

    ndone = 0
    while ndone < nsamples:
        xy = torch.tensor(rs.uniform(low=-3., high=3., size=(nsamples, 2)), dtype=torch.float)
        likelihoods = lik_fn(input=xy)

        u = torch.tensor(rs.uniform(low=0., high=1.01 * mode, size=nsamples), dtype=torch.float)
        accept = xy[u > likelihoods] if complementary else xy[u <= likelihoods]
        if accept.shape[0] > 0:
            samples.append(accept)
            ndone += accept.shape[0]
    
    return torch.cat(samples)
