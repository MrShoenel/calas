import math
import torch
from torch import Tensor
from torch.func import jacrev
from torch.nn import functional as F
from torch.distributions.normal import Normal
from .flow import CalasFlow
from .func import normal_cdf, normal_ppf_safe
from types import TracebackType
from typing import Generic, TypeVar, Self, Optional, Literal, override, final, Callable
from abc import ABC, abstractmethod
import numpy as np
from enum import Enum, StrEnum


T = TypeVar(name='T', bound=CalasFlow)

@final
class Space(StrEnum):
    Data = 'X'
    """
    The original data/input space."""

    Embedded = 'E'
    """
    The space after which the original data/input is now embedded (e.g., after a
    representation). Usually the result of going from X to E or back from B to E
    (the flow's inverse pass)."""

    Base = 'B'
    """
    The base space of the normalizing flow (defined by the chosen base distribution).
    Usually the result of going from E to B (the flow's forward pass)."""

    Quantiles = 'Q'
    """
    Quantiles of the data in base space, i.e., ppf(b)."""


@final
class Likelihood(StrEnum):
    """
    Enumeration of actions that indicate how some permutation should change the
    current likelihood of a sample.
    """
    Increase = '+'
    Decrease = '-'
    Randomize = '~'


@final
class LocsScales(Enum):
    Individual = 1
    """
    Computes the means and standard deviations individually for each sample in
    a batch."""

    Averaged = 2
    """
    Computes the means and standard deviation over an entire batch and averages
    the results."""

 



class Linear():

    @staticmethod
    def _icdf_safe(q: Tensor, tol: float=1e-7) -> Tensor:
        if q.dtype.itemsize < 4:
            tol = 1e-2
        return torch.clip(input=q, min=tol, max=1.-tol)
    

    def __init__(self, flow: T, space_in: Space, space_out: Space):
        super().__init__(flow=flow)

        self.space_in = space_in
        self.space_out = space_out
        
        self.normals = { clz: Normal(
            loc=self.flow.mean_for_class(clazz=clz).repeat(self.flow.num_dims),
            scale=torch.exp(self.flow.log_scale_for_class(clazz=clz).repeat(self.flow.num_dims))) for clz in range(flow.num_classes) }
    

    def modify3(self, batch: Tensor, classes: Tensor, target_lik: float, condition: Literal['concentrate', 'lower_than', 'greater_than'], concentrate_dim_log_tol: float=1.0, return_all: bool=False, max_steps: int=20, u_min: float=0.02, u_max=0.4) -> Tensor:
        """
        Linear modification always happens in the embedding space, as there are
        no guarantees at this point that the reconstruction is faithful. That
        means that the modified batch is always returned in the embedding space.
        """
        flow = self.flow
        assert isinstance(batch, Tensor) and batch.dim() == 2 and batch.shape[0] > 0, 'sample needs to be a non-empty 2D Tensor'
        if self.space_in == 'X': # This here method only operates in E space!
            batch = flow.X_to_E(input=batch)
        clz_int = classes.squeeze().to(dtype=torch.int64, device=batch.device)
        batch_shape = batch.shape
        embedded = batch.clone().detach()
        liks = flow.log_rel_lik_E(embeddings=embedded, classes=clz_int)


        results: list[Tensor] = []
        results_classes: list[Tensor] = []
        steps = 0
        while steps < max_steps and embedded.shape[0] > 0:
            steps += 1
            b_batch, b_batch_logdet = flow.E_to_B(embeddings=embedded, classes=clz_int)
            u = u_min + torch.rand_like(b_batch) * (u_max - u_min)
            
            q = torch.vstack(tensors=list(
                self.normals[clz_int[idx].item()].cdf(b_batch[idx]) for idx in range(b_batch.shape[0])))
            
            lamb = torch.where(q < 0.5, 1., -1.).to(dtype=q.dtype)
            lamb *= 1. if condition == 'greater_than' else -1.
            
            q_prime = q + lamb * u
            q_prime = torch.where((q_prime < .1) | (q_prime > .99), q, q_prime)
            
            b_prime = torch.vstack(tensors=list(
                self.normals[clz_int[idx].item()].icdf(Linear._icdf_safe(
                    q=q_prime[idx])) for idx in range(b_batch.shape[0])))
            
            embedded_prime = self.flow.E_from_B(base=b_prime, classes=clz_int)[0]
            b_prime_logdet = flow.E_to_B(embeddings=embedded_prime, classes=clz_int)[1]
            embedded_prime_liks = self.flow.log_rel_lik_E(embeddings=embedded_prime, classes=clz_int)


            # First replace those embeddings that are now 'better' (in terms of the condition)
            mask_replace = torch.where(embedded_prime_liks < liks, True, False) if condition == 'lower_than' else torch.where(embedded_prime_liks > liks, True, False)

            if torch.any(mask_replace).item():
                liks[mask_replace] = embedded_prime_liks[mask_replace]
                embedded[mask_replace] = embedded_prime[mask_replace]


            mask_accept = torch.where(embedded_prime_liks < target_lik, True, False) if condition == 'lower_than' else torch.where(embedded_prime_liks > target_lik, True, False)

            done = embedded_prime[mask_accept]
            if done.shape[0] > 0:
                results.append(done)
                results_classes.append(clz_int[mask_accept])
            clz_int = clz_int[~mask_accept]
            liks = liks[~mask_accept]
            embedded = embedded[~mask_accept]

        
        return torch.vstack(tensors=results), torch.cat(results_classes)



    

    def log_rel_lik_emb_unagg(self, embeddings: Tensor, clz: Tensor) -> tuple[Tensor, Tensor]:
        b, b_log_det = self.flow.E_to_B(embeddings=embeddings, classes=clz)
        return torch.vstack(tensors=list(
            self.normals[clz[idx].item()].log_prob(value=b[idx]) for idx in range(embeddings.shape[0])
        )), (b_log_det / b.shape[1]).unsqueeze(dim=0).T.repeat(1, b.shape[1])

    

    def modify2(self, batch: Tensor, classes: Tensor, target_lik: float, condition: Literal['concentrate', 'lower_than', 'greater_than'], concentrate_dim_log_tol: float=1.0, perc_change: float=0.003, return_all: bool=False) -> Tensor:
        """
        Linear modification always happens in the embedding space, as there are
        no guarantees at this point that the reconstruction is faithful. That
        means that the modified batch is always returned in the embedding space.
        """
        flow = self.flow
        assert isinstance(batch, Tensor) and batch.dim() == 2 and batch.shape[0] > 0, 'sample needs to be a non-empty 2D Tensor'
        if self.space_in == Space.Data: # This here method only operates in E space!
            batch = flow.X_to_E(input=batch)
        clz_int = classes.squeeze().to(dtype=torch.int64, device=batch.device)
        batch_shape = batch.shape
        embedded = batch.clone().detach()

        results_unacc: list[Tensor] = []
        results: list[Tensor] = []
        steps = 0
        while steps < 20 and embedded.shape[0] > 0:
            steps += 1
            b_batch = flow.E_to_B(embeddings=embedded, classes=clz_int)[0]
            u = 1e-5 + torch.rand_like(b_batch) * perc_change # ~(0, 0.3]% change/step
            
            q = torch.vstack(tensors=list(
                self.normals[clz_int[idx].item()].cdf(b_batch[idx]) for idx in range(b_batch.shape[0])))
            
            if condition == 'lower_than':
                # Check if the quantiles already reside at the extremes and skip those samples,
                # as we cannot make them worse.
                mask_exclude = torch.where((q.min(dim=1).values < 1e-7) | (q.max(dim=1).values > 1.-1e-7), True, False)
                if torch.any(mask_exclude).item():
                    results_unacc.append(embedded[mask_exclude])
                    embedded = embedded[~mask_exclude]
                    clz_int = clz_int[~mask_exclude]
                    continue
            
            # The following lambda tells us about which side of the normal distr.
            # we're on. One could also say that it indicates the sign of the gradient.
            lamb = torch.where(q < 0.5, 1., -1.).to(dtype=q.dtype)
            if condition == 'concentrate':
                b_batch_unagg_likelihood, b_batch_unagg_log_det =\
                    self.log_rel_lik_emb_unagg(embeddings=embedded, clz=clz_int)
                b_batch_unagg_likelihood += b_batch_unagg_log_det
                lamb *= torch.sign(input=(target_lik / b_batch.shape[1]) - b_batch_unagg_likelihood)
            else:
                lamb *= 1. if condition == 'greater_than' else -1.
            
            q_prime = q + lamb * u
            # Modify b according to the condition and target.
            b_prime = torch.vstack(tensors=list(
                self.normals[clz_int[idx].item()].icdf(Linear._icdf_safe(
                    q=q_prime[idx])) for idx in range(b_batch.shape[0])))
            
            # Now we inverse and forward the modified base and check the resulting likelihoods!
            # TODO: Check if the following is double work!
            embedded_prime, eld = self.flow.E_from_B(base=b_prime, classes=clz_int)
            embedded_prime_liks = self.flow.log_rel_lik_E(embeddings=embedded_prime, classes=clz_int)

            mask_accept: Tensor = None
            if condition == 'concentrate':
                # 'concentrate_dim_log_tol' is per dimension, so we got to sum this up.
                mask_accept = torch.where(torch.abs(embedded_prime_liks - target_lik) < concentrate_dim_log_tol, True, False)
            else:
                mask_accept = torch.where(embedded_prime_liks < target_lik, True, False) if condition == 'lower_than' else torch.where(embedded_prime_liks > target_lik, True, False)
            
            done = embedded_prime[mask_accept]
            if done.shape[0] > 0:
                results.append(done)
            clz_int = clz_int[~mask_accept]
            embedded = embedded_prime[~mask_accept]
        
        n_results = lambda: sum(r.shape[0] for r in results)
        if n_results() < batch_shape[0]:
            if return_all:
                results.append(embedded_prime) # Return even those that were not accepted
            if n_results() == 0:
                return torch.empty((0, batch_shape[1]))
        return torch.vstack(tensors=results)




    def _modify(self, sample: Tensor, sample_is_embedding: bool, classes: Tensor, target_lik: float, condition: Literal['concentrate', 'lower_than', 'greater_than'], concentrate_dim_log_tol: float=1.0, perc_change: float=0.003, return_all: bool=False) -> Tensor:
        flow = self.flow
        org_shape = sample.shape
        assert isinstance(sample, Tensor) and sample.dim() == 2 and sample.shape[0] > 1, 'sample needs to be a 2D Tensor with two or more data points.'
        if not sample_is_embedding:
            assert flow.can_reconstruct, 'Can only modify samples if they can be reconstructed'
        clz_int = classes.squeeze().to(dtype=torch.int64, device=sample.device)
        
        
        normals = { clz: Normal(
            loc=flow.mean_for_class(clazz=clz).repeat(flow.num_dims),
            scale=torch.exp(flow.log_scale_for_class(clazz=clz).repeat(flow.num_dims))) for clz in range(flow.num_classes) }


        def to_b(t: Tensor, clz: Tensor) -> tuple[Tensor, Tensor]:
            if not sample_is_embedding:
                t = flow.X_to_E(input=t)
            b, ld = flow.E_to_B(embeddings=t, classes=clz)
            return b, ld
        
        def from_b(b: Tensor, clz: Tensor) -> Tensor:
            t = flow.E_from_B(base=b, classes=clz)[0]
            if not sample_is_embedding:
                t = flow.X_from_E(embeddings=t)
            return t
        
        def lik(t: Tensor, clz: Tensor) -> Tensor:
            return (flow.log_rel_lik_E(embeddings=t, classes=clz) if sample_is_embedding else flow.log_rel_lik(input=t, classes=clz))
        
        def lik_unagg(t: Tensor, clz: Tensor) -> tuple[Tensor, Tensor]:
            if not sample_is_embedding:
                t = flow.X_to_E(input=t)
            b, b_log_det = flow.E_to_B(embeddings=t, classes=clz)

            return torch.vstack(tensors=list(
                normals[clz_int[idx].item()].log_prob(value=b[idx]) for idx in range(t.shape[0])
            )), (b_log_det / b.shape[1]).unsqueeze(dim=0).T.repeat(1, b.shape[1])

        
        results_unacc: list[Tensor] = []
        results: list[Tensor] = []
        sample = sample.clone().detach() # We make a copy now and chip-off acceptable results! The shape[0] will decrease!
        steps = 0
        while steps < 20 and sample.shape[0] > 0:
            steps += 1
            b_batch = to_b(t=sample, clz=clz_int)[0]
            u = 1e-5 + torch.rand_like(b_batch) * perc_change # ~(0, 0.3]% change/step

            q = torch.vstack(tensors=list(
                normals[clz_int[idx].item()].cdf(b_batch[idx]) for idx in range(b_batch.shape[0])))
            if condition == 'lower_than':
                # Check if the quantiles already reside at the extremes and skip those samples,
                # as we cannot make them worse.
                mask_exclude = torch.where((q.min(dim=1).values < 1e-7) | (q.max(dim=1).values > 1.-1e-7), True, False)
                if torch.any(mask_exclude).item():
                    results_unacc.append(sample[mask_exclude])
                    sample = sample[~mask_exclude]
                    clz_int = clz_int[~mask_exclude]
                    continue
            
            # The following lambda tells us about which side of the normal distr.
            # we're on. One could also say that it indicates the sign of the gradient.
            lamb = torch.where(q < 0.5, 1., -1.).to(dtype=q.dtype)
            if condition == 'concentrate':
                b_batch_unagg_likelihood, b_batch_unagg_log_det = lik_unagg(t=sample, clz=clz_int)
                b_batch_unagg_likelihood += b_batch_unagg_log_det
                lamb *= torch.sign(input=(target_lik / b_batch.shape[1]) - b_batch_unagg_likelihood)
            else:
                lamb *= 1. if condition == 'greater_than' else -1.
            
            q_prime = q + lamb * u
            # Modify b according to the condition and target.
            b_prime = torch.vstack(tensors=list(
                normals[clz_int[idx].item()].icdf(Linear._icdf_safe(
                    q=q_prime[idx])) for idx in range(b_batch.shape[0])))
            
            # Now we inverse and forward the modified and check the resulting likelihoods!
            samp_prime = from_b(b=b_prime, clz=clz_int)
            samp_prime_liks = lik(t=samp_prime, clz=clz_int)

            mask_accept: Tensor = None
            if condition == 'concentrate':
                # 'concentrate_dim_log_tol' is per dimension, so we got to sum this up.
                mask_accept = torch.where(torch.abs(samp_prime_liks - target_lik) < concentrate_dim_log_tol, True, False)
            else:
                mask_accept = torch.where(samp_prime_liks < target_lik, True, False) if condition == 'lower_than' else torch.where(samp_prime_liks > target_lik, True, False)
            
            done = samp_prime[mask_accept]
            if done.shape[0] > 0:
                results.append(done)
            clz_int = clz_int[~mask_accept]
            sample = samp_prime[~mask_accept]
        
        n_results = lambda: sum(r.shape[0] for r in results)
        if n_results() < org_shape[0]:
            if return_all:
                results.append(samp_prime) # Return even those that were not accepted
            if n_results() == 0:
                return torch.empty((0, org_shape[1]))
        return torch.vstack(tensors=results)
        


    def modify(self, sample: Tensor, classes: Tensor, target_lik: float, condition: Literal['concentrate', 'lower_than', 'greater_than'], concentrate_dim_log_tol: float=1.0, perc_change: float=0.003, return_all: bool=False) -> Tensor:
        return self._modify(sample=sample, sample_is_embedding=False, classes=classes, target_lik=target_lik, condition=condition, concentrate_dim_log_tol=concentrate_dim_log_tol, perc_change=perc_change, return_all=return_all)


    def modify_emb(self, sample: Tensor, classes: Tensor, target_lik: float, condition: Literal['concentrate', 'lower_than', 'greater_than'], concentrate_dim_log_tol: float=1.0, perc_change: float=0.003, return_all: bool=False) -> Tensor:
        return self._modify(sample=sample, sample_is_embedding=True, classes=classes, target_lik=target_lik, condition=condition, concentrate_dim_log_tol=concentrate_dim_log_tol, perc_change=perc_change, return_all=return_all)







class Permute(Generic[T], ABC):
    """
    This is the base class for all permutations that can be applied to a sample
    in order to change it, such that it incurs a lower or higher loss.
    """
    
    def __init__(self, flow: T, u_min: float=0.01, u_max: float=0.01, u_frac_negative: float=0.0, seed: Optional[int]=0):
        self.flow = flow
        self.u_min = u_min
        self.u_max = u_max
        self.u_frac_negative = u_frac_negative
        self.gen = np.random.default_rng(seed=seed)
    

    def __enter__(self) -> Self:
        self.was_training = self.flow.training
        self.flow.eval()
        self.ng = torch.no_grad()
        self.ng.__enter__()
        return self
    
    def __exit__(self, exc_type: Optional[type[BaseException]]=None, exc_value: Optional[BaseException]=None, traceback: Optional[TracebackType]=None) -> None:
        self.ng.__exit__(exc_type=exc_type, exc_value=exc_value, traceback=traceback)
        if self.was_training:
            self.flow.train()
    

    @property
    @abstractmethod
    def space_in(self) -> Space: ... # pragma: no cover

    @property
    @abstractmethod
    def space_out(self) -> Space: ... # pragma: no cover
    

    @abstractmethod
    def permute(self, batch: Tensor, classes: Tensor, likelihood: Likelihood) -> Tensor: ... # pragma: no cover
    

    def u_like(self, input: Tensor) -> Tensor:
        u = self.u_min + torch.tensor(data=self.gen.random(size=input.shape), device=input.device, dtype=input.dtype) * (self.u_max - self.u_min)
        if self.u_frac_negative > 0.:
            assert self.u_frac_negative <= 1.
            neg = torch.tensor(data=self.gen.random(size=u.shape), device=u.device, dtype=u.dtype)
            u *= torch.where(neg <= self.u_frac_negative, -1., 1.)
        return u


class PermuteData(Permute[T], ABC):
    pass


class Dist2Dist(Permute[T], ABC):
    """
    The base-class for permutations that change the data's underlying distribution
    while simultaneously preserving some or all of the data's properties and/or
    structure.
    """

    def __init__(self, flow: T, u_min: float=0.01, u_max: float=0.01, u_frac_negative: float=0.0, locs_scales_mode: LocsScales=LocsScales.Individual, locs_scales_flow_frac: float=0.2, seed: Optional[int]=0):
        super().__init__(flow=flow, seed=seed, u_min=u_min, u_max=u_max, u_frac_negative=u_frac_negative)
        assert locs_scales_flow_frac >= 0.0 and locs_scales_flow_frac <= 1.0
        self.u_min = u_min
        self.u_max = u_max
        self.u_frac_negative = u_frac_negative
        self.locs_scales_mode = locs_scales_mode
        self.locs_scales_flow_frac = locs_scales_flow_frac
    

    def flow_locs(self, classes: Tensor) -> Tensor:
        clz_int: Tensor = torch.atleast_1d(classes.squeeze().to(torch.int64))
        return torch.vstack(tensors=list(
            self.flow.mean_for_class(clazz=clz_int[idx].item()) for idx in range(clz_int.numel()))).repeat(1, self.flow.num_dims)
    

    def flow_scales(self, classes: Tensor) -> Tensor:
        clz_int: Tensor = torch.atleast_1d(classes.squeeze().to(torch.int64))
        return torch.exp(torch.vstack(tensors=list(
            self.flow.log_scale_for_class(clazz=clz_int[idx].item()) for idx in range(clz_int.numel()))).repeat(1, self.flow.num_dims))


    def flow_locs_and_scales(self, classes: Tensor) -> tuple[Tensor, Tensor]:
        return self.flow_locs(classes=classes), self.flow_scales(classes=classes)
    

    def batch_locs_and_scales(self, batch: Tensor) -> tuple[Tensor, Tensor]:
        means: Tensor = None
        stds: Tensor = None
        if self.locs_scales_mode == LocsScales.Individual:
            means = batch.mean(dim=1).unsqueeze(-1).repeat(1, self.flow.num_dims)
            stds = batch.std(dim=1).unsqueeze(-1).repeat(1, self.flow.num_dims)
        else:
            means = batch.mean().repeat(*batch.shape)
            stds = batch.std().repeat(*batch.shape)
        return means, stds
    

    def averaged_locs_and_scales(self, batch: Tensor, classes: Optional[Tensor]=None) -> tuple[Tensor, Tensor]:
        if self.locs_scales_flow_frac == 0.0:
            return self.batch_locs_and_scales(batch=batch)
        elif self.locs_scales_flow_frac == 1.0:
            assert isinstance(classes, Tensor)
            return self.flow_locs_and_scales(classes=classes)
        
        # else:
        assert isinstance(classes, Tensor)
        batch_means, batch_stds = self.batch_locs_and_scales(batch=batch)
        flow_means, flow_stds = self.flow_locs_and_scales(classes=classes)
        return self.locs_scales_flow_frac * flow_means + (1.0 - self.locs_scales_flow_frac) * batch_means,\
            self.locs_scales_flow_frac * flow_stds + (1.0 - self.locs_scales_flow_frac) * batch_stds



class Normal2Normal(Dist2Dist[T]):
    """
    Takes the data in B-space and assumes it is normal. Calculates the data's
    mean and std and then uses this transform the data to quantiles and then
    to real data again using another normal distribution that is more or less
    likely than the flow's base distribution.

    Note: This is a "sample-wise" permutation. Each sample is transposed in B
    and then modified.
    """

    def __init__(self, flow: T, seed: Optional[int]=0):
        super().__init__(flow=flow, seed=seed)
        
        self.normals = { clz: Normal(
            loc=self.flow.mean_for_class(clazz=clz).repeat(self.flow.num_dims),
            scale=torch.exp(self.flow.log_scale_for_class(clazz=clz).repeat(self.flow.num_dims))) for clz in range(flow.num_classes) }
    
    
    @override
    def permute(self, embeddings: Tensor, classes: Tensor, likelihood: Literal['increase', 'decrease', 'random']) -> Tensor:
        clz_int = torch.atleast_1d(classes.squeeze().to(torch.int64))
        b = self.flow.E_to_B(embeddings=embeddings, classes=classes)[0]
        
        flow_means = torch.vstack(tensors=list(
            self.normals[clz_int[idx].item()].loc for idx in range(b.shape[0])))
        
        b_means = b.mean(dim=1).unsqueeze(-1).repeat(1, self.flow.num_dims)
        b_stds = b.std(dim=1).unsqueeze(-1).repeat(1, self.flow.num_dims)

        # If we want to increase the likelihood, the current means of the data
        # in b need to approach the flow's means. The scale needs to decrease
        # for the data to become more likely.

        means_diff = flow_means - b_means
        # The sign tells us towards which way the means get closer
        means_sign = torch.sign(means_diff)

        if likelihood == 'decrease':
            means_sign *= -1.
        elif likelihood == 'random':
            r = torch.tensor(self.gen.random(size=means_sign.shape), device=means_sign.device)
            means_sign = torch.where(r < 0.5, -1., 1.)

        # Let's take a small step towards or away from the mean:
        altered_means = b_means + means_sign * b_stds[:, 0:1] * .1

        stds_mult = torch.full_like(b_stds, fill_value=0.9 if likelihood == 'increase' else 1.1)
        if likelihood == 'random':
            r = torch.tensor(self.gen.random(size=stds_mult.shape), device=stds_mult.device)
            stds_mult = torch.where(r < 0.5, 0.9, 1.1)
        
        altered_stds = b_stds * stds_mult

        q = torch.vmap(normal_cdf)(b, b_means, b_stds)
        b_prime = torch.vmap(normal_ppf_safe)(q, altered_means, altered_stds)

        return self.flow.E_from_B(base=b_prime, classes=clz_int)



class Data2Data_Grad(PermuteData[T]):
    """
    A same-space permutation that updates the input using the gradient with
    regard to the appropriate loss function.
    """
    def __init__(self, flow: T, space: Space, u_min: float=0.01, u_max: float=0.01, scale_grad: bool=True, seed: Optional[int]=0):
        super().__init__(flow=flow, seed=seed)
        if space == Space.Quantiles:
            raise Exception(f'Space {space} is not supported.')
        self.space = space
        self.u_min = u_min
        self.u_max = u_max
        self.scale_grad = scale_grad
    
    @override
    @property
    def space_in(self):
        return self.space
    
    @property
    @override
    def space_out(self):
        return self.space
    
    @override
    def permute(self, batch: Tensor, classes: Tensor, likelihood: Likelihood) -> Tensor:
        if likelihood == Likelihood.Randomize:
            raise Exception('This permutation explicitly leverages gradient information. If you require a random permutation, try another class.')
        
        clz_int = torch.atleast_1d(classes.squeeze().to(torch.int64))
        # 'batch' could be in any of these 3 spaces.
        loss_fn_grad = self.flow.loss_wrt_X_grad if self.space == Space.Data else (self.flow.loss_wrt_E_grad if self.space == Space.Embedded else self.flow.loss_wrt_B_grad)

        grad = loss_fn_grad(batch, clz_int)
        # The gradient points in the direction where the *loss* gets larger,
        # which is the same as the likelihood getting *lower*. So, we have to
        # conditionally inverse their relation as per default, it will decrease
        # the likelihood.
        grad_sign = torch.sign(grad)
        if likelihood == Likelihood.Increase:
            grad_sign *= -1.
        
        grad_weight = grad.reciprocal() / grad.reciprocal().max() if self.scale_grad else 1.
        u = self.u_like(input=batch)
        batch_prime = batch + grad_sign * grad_weight * u
        return batch_prime





class LinearPerm_in_B(Dist2Dist[T]):
    def __init__(self, flow: T, u_min: float=0.001, u_max: float=0.05, seed: Optional[int]=0):
        super().__init__(flow=flow, seed=seed)
        self.u_min = u_min
        self.u_max = u_max
    

    def permute(self, embeddings: Tensor, classes: Tensor, likelihood: Literal['increase', 'decrease', 'random']) -> Tensor:
        clz_int = torch.atleast_1d(classes.squeeze().to(torch.int64))
        b = self.flow.E_to_B(embeddings=embeddings, classes=classes)[0]

        
        flow_means = torch.vstack(tensors=list(
            self.flow.mean_for_class(clazz=clz_int[idx].item()) for idx in range(b.shape[0]))).repeat(1, self.flow.num_dims)
        flow_stds = torch.exp(torch.vstack(tensors=list(
            self.flow.log_scale_for_class(clazz=clz_int[idx].item()) for idx in range(b.shape[0]))).repeat(1, self.flow.num_dims))

        # For this synthesis method, it appears that using individual means and
        # stds works better than taking the whole batch into account.
        b_means = b.mean(dim=1).unsqueeze(-1).repeat(1, self.flow.num_dims)
        b_stds = b.std(dim=1).unsqueeze(-1).repeat(1, self.flow.num_dims)
        # b_means = b.mean().repeat(*b.shape)
        # b_stds = b.std().repeat(*b.shape)

        use_means = 0.2 * flow_means + 0.8 * b_means
        use_stds = 0.2 * flow_stds + 0.8 * b_stds

        u = self.u_min + torch.tensor(data=self.gen.random(size=b.shape), device=b.device, dtype=b.dtype) * (self.u_max - self.u_min)
        q: Tensor = torch.vmap(normal_cdf)(b, use_means, use_stds)
        lamb = torch.where(q < 0.5, 1., -1.).to(dtype=q.dtype)
        if likelihood == 'decrease':
            lamb *= -1
        q_prime = q + lamb * u

        b_prime = torch.vmap(normal_ppf_safe)(q_prime, use_means, use_stds)
        return self.flow.E_from_B(base=b_prime, classes=clz_int)



class Normal2Normal_Grad(Dist2Dist[T]):
    """
    Changes the distribution of one or more samples using one of three methods. In
    either case we assume each sample follows an approximate normal distribution
    (in B; this permutation is Base-2-Base), which should be the case for
    somewhat or well-trained normalizing flow. In any case, this permutation
    averages the means and scales of the flow's base distribution with those of
    the sample(s).
    NOTE: This permutation leverages gradient information to determine how the
    parameters under the chosen method should be altered. This makes this method
    inherently more computationally expensive. Therefore, consider also
    :code:`Normal2Normal_Linear`.

    The first method, 'loc_scale', alters the sample's location and scale, while
    the quantiles stay the same. Simply put, it is like serializing the sample
    where it currently is and deserializing where it should be using optimized
    location and scale.

    The second method, 'quantiles', alters the quantiles of a sample by moving
    all of its dimensions simultaneously under the assumed mixture. Since `u` is
    usually drawn from a uniform distribution, the sample's empirical distribution
    stays approximately the same.

    The third method, 'hybrid', alters the sample's quantiles using the gradient
    of the sample itself. This works because B and Q have the same dimensionality
    and because under a normal distribution, there is a strong monotonic relation
    between both spaces. This is a somewhat special case; usually you would want
    the method 'quantiles' instead.
    """
    def __init__(self, flow: T, method: Literal['loc_scale', 'quantiles', 'hybrid'], u_min: float=0.001, u_max: float=0.05, u_frac_negative: float=0.0, scale_grad: bool=True, seed: Optional[int]=0):
        super().__init__(flow=flow, seed=seed, u_min=u_min, u_max=u_max, u_frac_negative=u_frac_negative)
        self.method = method
        self.u_min = u_min
        self.u_max = u_max
        self.scale_grad = scale_grad
    

    @property
    @override
    def space_in(self) -> Space:
        return Space.Base
    

    @property
    @override
    def space_out(self) -> Space:
        return Space.Base
    

    def permute(self, batch: Tensor, classes: Tensor, likelihood: Likelihood):
        if likelihood == Likelihood.Randomize:
            raise Exception(f'{Normal2Normal_Grad.__name__} leverages gradient information to perform a distributional permutation, such that using likelihood=`{Likelihood.Randomize}` would defeat this purpose. However, in some cases it might still make sense to randomly take steps in the (undesired) direction. For that, it is recommended to set the general direction with `likelihood` and to then pick `u_frac_negative` to be smaller than `0.5`. If you want true randomized likelihood, either set `u_frac_negative=0.5` or use another class that does not rely on the gradient, such as `{Linear.__name__}`.')
        clz_int = torch.atleast_1d(classes.squeeze().to(torch.int64))
        b = batch # The given data must be in B-space for these permutations!

        flow_means, flow_stds = self.flow_locs_and_scales(classes=classes)

        b_means = b.mean(dim=1).unsqueeze(-1).repeat(1, self.flow.num_dims)
        b_stds = b.std(dim=1).unsqueeze(-1).repeat(1, self.flow.num_dims)

        use_means = 0.2 * flow_means + 0.8 * b_means
        use_stds = 0.2 * flow_stds + 0.8 * b_stds
        u = self.u_like(input=b)

        if self.method == 'hybrid':
            return self.permute_hybrid(base=b, u=u, use_means=use_means, use_stds=use_stds, classes=classes, likelihood=likelihood)
        elif self.method == 'quantiles':
            return self.permute_quantiles(base=b, u=u, use_means=use_means, use_stds=use_stds, classes=clz_int, likelihood=likelihood)
        
        return self.permute_loc_scale(base=b, u=u, use_means=use_means, use_stds=use_stds, classes=clz_int, likelihood=likelihood)



    def permute_loc_scale(self, base: Tensor, u: Tensor, use_means: Tensor, use_stds: Tensor, classes: Tensor, likelihood: Likelihood) -> Tensor:
        """
        Permutes samples by varying the parameters of each sample's underyling (assumed)
        normal distribution. Given a sample in B, assumes it is normally distributed and
        computes the standard and mean of its transpose, :math:`b^\top`. Then, we take
        the derivative of the loss w.r.t. those means and stds. The permuted B is then
        generated by first mapping the original using its empirical means and stds to
        quantiles, and then taking those quantiles and the altered means and stds to
        re-generate it at the new location using the new scale.
        """
        q: Tensor = torch.vmap(normal_cdf)(base, use_means, use_stds)

        def loss_fn(mu: Tensor, sigma: Tensor) -> Tensor:
            return self.flow.loss_wrt_Q(q=q, loc=mu, scale=sigma, classes=classes)
        
        loss_fn_grad = jacrev(func=loss_fn, argnums=(0,1))
        means_grad, stds_grad = loss_fn_grad(use_means, use_stds)
        means_grad_sign, stds_grad_sign = torch.sign(input=means_grad), torch.sign(input=stds_grad)
        means_grad, stds_grad = torch.abs(means_grad), torch.abs(stds_grad)
        if likelihood == Likelihood.Increase:
            means_grad_sign *= -1.
            stds_grad_sign *= -1.
        
        means_grad_weight = means_grad.reciprocal() / means_grad.reciprocal().max() if self.scale_grad else 1.
        means_prime = use_means + means_grad_sign * means_grad_weight * u

        stds_grad_weight = stds_grad.reciprocal() / stds_grad.reciprocal().max() if self.scale_grad else 1.
        stds_prime = use_stds + stds_grad_sign * stds_grad_weight * u

        b_prime = torch.vmap(normal_ppf_safe)(q, means_prime, stds_prime)
        return b_prime


    def _permute_hybrid_or_quantiles(self, hybrid: bool, base: Tensor, u: Tensor, use_means: Tensor, use_stds: Tensor, classes: Tensor, likelihood: Likelihood) -> Tensor:
        loss_fn_grad: Callable[[Tensor], Tensor] = None
        if hybrid:
            loss_fn_grad = lambda b: self.flow.loss_wrt_B_grad(base=b, classes=classes)
        else:
            loss_fn_grad = jacrev(lambda q: self.flow.loss_wrt_Q(q=q, loc=use_means, scale=use_stds, classes=classes))

        q = torch.vmap(normal_cdf)(base, use_means, use_stds)
        grad = loss_fn_grad(base if hybrid else q)
        grad_sign = torch.sign(input=grad)
        grad = torch.abs(input=grad)
        grad_weight = grad.reciprocal() / grad.reciprocal().max() if self.scale_grad else 1.
        if likelihood == Likelihood.Increase:
            grad_sign *= -1.
        
        q_prime = q + grad_sign * grad_weight * u
        b_prime = torch.vmap(normal_ppf_safe)(q_prime, use_means, use_stds)
        return b_prime
    

    def permute_hybrid(self, base: Tensor, u: Tensor, use_means: Tensor, use_stds: Tensor, classes: Tensor, likelihood: Likelihood) -> Tensor:
        """
        Permutes samples similarly to :code:`permute_quantiles()`, but takes the gradient
        from the loss as incurred by B itself, not its quantiles.
        """
        return self._permute_hybrid_or_quantiles(hybrid=True, base=base, u=u, use_means=use_means, use_stds=use_stds, classes=classes, likelihood=likelihood)


    def permute_quantiles(self, base: Tensor, u: Tensor, use_means: Tensor, use_stds: Tensor, classes: Tensor, likelihood: Likelihood) -> Tensor:
        """
        Permutes samples by varying the quantiles under each sample's underlying (assumed)
        normal distribution. Given a (transposed) sample in B, this method uses its quantiles
        under a given mean and scale to produce an altered sample. We do so by taking the
        derivative of the loss function w.r.t. the quantiles. In other words, the gradient
        tells us how we need to modify the quantiles in order to change the sample.
        """
        return self._permute_hybrid_or_quantiles(hybrid=False, base=base, u=u, use_means=use_means, use_stds=use_stds, classes=classes, likelihood=likelihood)


class Normal2Normal_NoGrad(Dist2Dist[T]):
    def __init__(self, flow: T, method: Literal['loc_scale', 'quantiles', 'hybrid']=None, u_min: float=0.001, u_max: float=0.001, u_frac_negative: float=0.0, locs_scales_mode: LocsScales=LocsScales.Individual, locs_scales_flow_frac: float=0.2, stds_step_perc: float=0.025, seed: Optional[int]=0):
        super().__init__(flow=flow, seed=seed, u_min=u_min, u_max=u_max, u_frac_negative=u_frac_negative, locs_scales_mode=locs_scales_mode, locs_scales_flow_frac=locs_scales_flow_frac)
        self.stds_step_perc = stds_step_perc
    

    @property
    @override
    def space_in(self) -> Space:
        return Space.Base
    
    @property
    @override
    def space_out(self) -> Space:
        return Space.Base
    
    def permute(self, batch: Tensor, classes: Tensor, likelihood: Likelihood):
        """
        Approximate implementation of old  method of class in line 461
        """
        clz_int = torch.atleast_1d(classes.squeeze().to(torch.int64))

        use_means, use_stds = self.averaged_locs_and_scales(batch=batch, classes=clz_int)
        flow_means = self.flow_locs(classes=classes)
        batch_means = self.batch_locs_and_scales(batch=batch)[0]

        # If we want to increase the likelihood, the current means of the data
        # in b need to approach the flow's means. The scale needs to decrease
        # for the data to become more likely.

        means_diff = flow_means - batch_means
        # The sign tells us towards which way the means get closer.
        means_sign = torch.sign(means_diff)

        if likelihood == Likelihood.Decrease:
            means_sign *= -1.
        del batch_means
        
        # Here in this method, if we wanted to randomize the resulting likelihoods,
        # we would control it by setting up `u` to be exactly 1.0 with a frac of 0.5.
        # We could also choose 
        u = self.u_like(means_sign)

        # Let's take a small step that also incorporates the standard deviation
        # as found in the sample.
        means_prime = use_means + means_sign * use_stds[:, 0:1] * u
        
        # If we want to increase the likelihood, the SD should shrink.
        stds_mult: float
        if likelihood == Likelihood.Increase:
            stds_mult = 1.0 - self.stds_step_perc # for 5%/0.05, this'll be 0.95
        else:
            stds_mult = 1.0 + self.stds_step_perc # for 5%/0.05, this'll be 1.05
        
        stds_prime = use_stds * stds_mult
        use_means, use_stds = self.averaged_locs_and_scales(batch=batch, classes=clz_int)

        q = torch.vmap(normal_cdf)(batch, use_means, use_stds)
        batch_prime = torch.vmap(normal_ppf_safe)(q, means_prime, stds_prime)

        return batch_prime
