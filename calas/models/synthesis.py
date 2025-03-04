import math
import torch
from torch import Tensor
from torch.func import jacrev
from torch.nn import functional as F
from torch.distributions.normal import Normal
from .flow import CalasFlow
from .func import normal_cdf, normal_ppf_safe
from types import TracebackType
from typing import Generic, TypeVar, Self, Optional, Literal, override, final
from abc import ABC, abstractmethod
import numpy as np
from enum import StrEnum


T = TypeVar(name='T', bound=CalasFlow)

@final
class Space(StrEnum):
    Data = 'Data'
    Embedded = 'Embedded'
    Base = 'Base'


class Synthesis(Generic[T]):
    def __init__(self, flow: T):
        self.flow = flow
    

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



class RandomSynthesis(Synthesis[T]):

    def dist_2_dist(self, embeddings: Tensor, classes: Tensor) -> Tensor:
        b, log_det = self.flow.E_to_B(embeddings=embeddings, classes=classes)
        pass


    def modify_emb(self, embeddings: Tensor, classes: Tensor, target_lik: float, condition: Literal['less_than', 'greater_than'], max_steps: int=20, seed: Optional[int]=None) -> Tensor:
        gen = np.random.default_rng(seed=seed)
        flow = self.flow
        assert isinstance(embeddings, Tensor) and embeddings.dim() == 2 and embeddings.shape[0] > 0, 'sample needs to be a non-empty 2D Tensor'
        clz_int = classes.squeeze().to(dtype=torch.int64, device=embeddings.device)

        methods = [
            lambda emb, clz: self.dist_2_dist(embeddings=emb, classes=clz)
        ]

        liks = flow.log_rel_lik_emb(embeddings=embeddings, classes=classes)
        results: list[Tensor] = []
        results_classes: list[Tensor] = []
        steps = 0
        while steps < max_steps and embeddings.shape[0] > 0:
            steps += 1
            use_method = methods[gen.integers(len(methods))]
            



class Linear(Synthesis[T]):

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
        liks = flow.log_rel_lik_emb(embeddings=embedded, classes=clz_int)


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
            embedded_prime_liks = self.flow.log_rel_lik_emb(embeddings=embedded_prime, classes=clz_int)


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
            embedded_prime_liks = self.flow.log_rel_lik_emb(embeddings=embedded_prime, classes=clz_int)

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
            return (flow.log_rel_lik_emb(embeddings=t, classes=clz) if sample_is_embedding else flow.log_rel_lik(input=t, classes=clz))
        
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






class CurseOfDimAnomaly:
    pass



class Permute(Generic[T]):
    """
    This is the base class for all permutations that can be applied to a sample
    in order to change it, such that it incurs a lower or higher loss.
    """
    
    def __init__(self, flow: T, seed: Optional[int]=0):
        self.flow = flow
        self.gen = np.random.default_rng(seed=seed)
    

    @abstractmethod
    def permute(self, embeddings: Tensor, classes: Tensor, likelihood: Literal['increase', 'decrease', 'random']) -> Tensor: ... # pragma: no cover



class Dist2Dist(Permute[T], ABC):
    """
    The base-class for permutations that change the data's underlying distribution
    while simultaneously preserving some or all of the data's properties and/or
    structure.
    """



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


class GradientPerm_in_QofB(Dist2Dist[T]):
    def __init__(self, flow: T, u_min: float=0.01, u_max: float=0.01, scale_grad: bool=True, seed: Optional[int]=0):
        super().__init__(flow=flow, seed=seed)
        self.u_min = u_min
        self.u_max = u_max
        self.scale_grad = scale_grad
    

    def permute(self, embeddings: Tensor, classes: Tensor, likelihood) -> Tensor:
        clz_int = torch.atleast_1d(classes.squeeze().to(torch.int64))
        b = self.flow.E_to_B(embeddings=embeddings, classes=classes)[0]
        
        flow_means = torch.vstack(tensors=list(
            self.flow.mean_for_class(clazz=clz_int[idx].item()) for idx in range(b.shape[0]))).repeat(1, self.flow.num_dims)
        flow_stds = torch.exp(torch.vstack(tensors=list(
            self.flow.log_scale_for_class(clazz=clz_int[idx].item()) for idx in range(b.shape[0]))).repeat(1, self.flow.num_dims))

        b_means = b.mean(dim=1).unsqueeze(-1).repeat(1, self.flow.num_dims)
        b_stds = b.std(dim=1).unsqueeze(-1).repeat(1, self.flow.num_dims)

        use_means = 0.2 * flow_means + 0.8 * b_means
        use_stds = 0.2 * flow_stds + 0.8 * b_stds
        q: Tensor = torch.vmap(normal_cdf)(b, use_means, use_stds)

        def loss_fn(quantiles: Tensor) -> Tensor:
            b_prime: Tensor = torch.vmap(normal_ppf_safe)(quantiles, use_means, use_stds)
            return self.flow.loss_B(base=b_prime, classes=clz_int)
        
        loss_fn_grad = jacrev(func=loss_fn)
        q_grad = loss_fn_grad(q)
        q_grad_sign = torch.sign(q_grad)
        q_grad = torch.abs(q_grad)
        if likelihood == 'increase':
            q_grad_sign *= -1.
        
        u = self.u_min + torch.tensor(data=self.gen.random(size=b.shape), device=q.device, dtype=b.dtype) * (self.u_max - self.u_min)
        q_grad_weight = q_grad.reciprocal() / q_grad.reciprocal().max() if self.scale_grad else 1.
        q_prime = q + q_grad_sign * q_grad_weight * u

        b_prime = torch.vmap(normal_ppf_safe)(q_prime, use_means, use_stds)
        return b_prime


class GradientPerm_in_B(Dist2Dist[T]):

    def __init__(self, flow: T, u_min: float=0.01, u_max: float=0.01, scale_grad: bool=True, seed: Optional[int]=0):
        super().__init__(flow=flow, seed=seed)
        self.u_min = u_min
        self.u_max = u_max
        self.scale_grad = scale_grad
    

    def permute(self, embeddings, classes, likelihood):
        if likelihood == 'random':
            raise Exception("Just use a linear permutation without gradient, because 'random' just randomizes the gradient's sign.")
        
        clz_int = torch.atleast_1d(classes.squeeze().to(torch.int64))
        b = self.flow.E_to_B(embeddings=embeddings, classes=classes)[0]

        grad = self.flow.loss_grad_wrt_B(base=b, classes=classes)
        # The gradient points in the direction where the *loss* gets larger,
        # which is the same as the likelihood getting *lower*. So, we have to
        # conditionally inverse their relation as per default, it will decrease
        # the likelihood.
        grad_sign = torch.sign(input=grad)
        grad = torch.abs(grad)
        if likelihood == 'increase':
            grad_sign *= -1.

        # One big question is which distribution we should assume for the current
        # data. Ideally, we want to use the Flow's base distribution. However, that
        # is not always practical and especially problematic if the data in the base
        # space is too far away. Because then, the quantiles will be too extreme and
        # difficult or impossible to manipulate. Therefore, as a practical solution,
        # it works quite well to assume the empirical distribution of each sample and
        # manipulate its quantiles there. Another solution is to use the average of
        # the flow's distribution's means/std and the data's. However, that can also
        # result in large jumps of the likelihood due to still too-extreme quantiles.
        # Therefore, I recommend to only manipulate the quantiles under the sample's
        # empirical distribution.

        flow_means = torch.vstack(tensors=list(
            self.flow.mean_for_class(clazz=clz_int[idx].item()) for idx in range(b.shape[0]))).repeat(1, self.flow.num_dims)
        flow_stds = torch.exp(torch.vstack(tensors=list(
            self.flow.log_scale_for_class(clazz=clz_int[idx].item()) for idx in range(b.shape[0]))).repeat(1, self.flow.num_dims))
        
        # Another thing: When using empirical distributions, it actually works better
        # to take mean and std from the entire batch, not on a per-sample basis. It
        # usually results in more fine-grained steps.
        # The following are per-sample mean & std:
        b_means = b.mean(dim=1).unsqueeze(-1).repeat(1, self.flow.num_dims)
        b_stds = b.std(dim=1).unsqueeze(-1).repeat(1, self.flow.num_dims)

        use_means = 0.2 * flow_means + 0.8 * b_means
        use_stds = 0.2 * flow_stds + 0.8 * b_stds
        q = torch.vmap(normal_cdf)(b, use_means, use_stds)

        # Let's use empirical means and stds; I kept the other versions for reference.
        # TODO: We should perhaps encapsulate all this into a method and make it
        # configurable.
        # The following also works really well:
        # b_means = b.mean().repeat(*b.shape)
        # b_stds = b.std().repeat(*b.shape)
        # use_means = b_means
        # use_stds = b_stds
        
        u = self.u_min + torch.tensor(data=self.gen.random(size=b.shape), device=b.device, dtype=b.dtype) * (self.u_max - self.u_min)
        # TODO: Instead of softmax,reciprocal we should perhaps just use the reciprocal
        # and normalize it such that the greatest weight is 1
        # q_prime = q + grad_sign * u * (F.softmax(torch.abs(grad).reciprocal()) if self.scale_grad else 1.)
        grad_weight = grad.reciprocal() / grad.reciprocal().max() if self.scale_grad else 1.
        q_prime = q + grad_sign * grad_weight * u

        b_prime = torch.vmap(normal_ppf_safe)(q_prime, use_means, use_stds)
        return self.flow.E_from_B(base=b_prime, classes=clz_int)


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


class Normal2NormalGrad(Dist2Dist[T]):
    """
    Permutes samples by varying the parameters of each sample's underyling (assumed)
    normal distribution. Given a sample in B, assumes it is normally distributed and
    computes the standard and mean of its transpose, :math:`b^\top`. Then, we take
    the derivative of the loss w.r.t. those means and stds. The permuted B is then
    generated by first mapping the original using its empirical means and stds to
    quantiles, and then taking those quantiles and the altered means and stds to
    re-generate it at the new location using the new scale.
    """
    def __init__(self, flow: T, u_min: float=0.001, u_max: float=0.05, seed: Optional[int]=0):
        super().__init__(flow=flow, seed=seed)
        self.u_min = u_min
        self.u_max = u_max
    

    def permute(self, embeddings, classes, likelihood):
        clz_int = torch.atleast_1d(classes.squeeze().to(torch.int64))
        b = self.flow.E_to_B(embeddings=embeddings, classes=classes)[0]
        
        flow_means = torch.vstack(tensors=list(
            self.flow.mean_for_class(clazz=clz_int[idx].item()) for idx in range(b.shape[0]))).repeat(1, self.flow.num_dims)
        flow_stds = torch.exp(torch.vstack(tensors=list(
            self.flow.log_scale_for_class(clazz=clz_int[idx].item()) for idx in range(b.shape[0]))).repeat(1, self.flow.num_dims))

        b_means = b.mean(dim=1).unsqueeze(-1).repeat(1, self.flow.num_dims)
        b_stds = b.std(dim=1).unsqueeze(-1).repeat(1, self.flow.num_dims)

        use_means = 0.2 * flow_means + 0.8 * b_means
        use_stds = 0.2 * flow_stds + 0.8 * b_stds
        q: Tensor = torch.vmap(normal_cdf)(b, use_means, use_stds)

        def loss_fn(mu: Tensor, sigma: Tensor) -> Tensor:
            b_prime: Tensor = torch.vmap(normal_ppf_safe)(q, mu, sigma)
            return self.flow.loss_B(base=b_prime, classes=clz_int)
        
        loss_fn_grad = jacrev(func=loss_fn, argnums=(0,1))
        means_grad, stds_grad = loss_fn_grad(use_means, use_stds)
        means_grad_sign, stds_grad_sign = torch.sign(input=means_grad), torch.sign(input=stds_grad)
        means_grad, stds_grad = torch.abs(means_grad), torch.abs(stds_grad)
        if likelihood == 'increase':
            means_grad_sign *= -1.
            stds_grad_sign *= -1.
        
        # We'll use the same `u` for means and stds.
        u = self.u_min + torch.tensor(data=self.gen.random(size=b.shape), device=b.device, dtype=b.dtype) * (self.u_max - self.u_min)
        means_grad_weight = means_grad.reciprocal() / means_grad.reciprocal().max()
        means_prime = use_means + means_grad_sign * means_grad_weight * u

        stds_grad_weight = stds_grad.reciprocal() / stds_grad.reciprocal().max()
        stds_prime = use_stds + stds_grad_sign * stds_grad_weight * u

        b_prime = torch.vmap(normal_ppf_safe)(q, means_prime, stds_prime)
        return b_prime
