import torch
from torch import Tensor
from torch.distributions.normal import Normal
from .flow import CalasFlow
from types import TracebackType
from typing import Generic, TypeVar, Self, Optional, Literal


T = TypeVar(name='T', bound=CalasFlow)
Space = Literal['X', 'E']


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
    

    def modify2(self, batch: Tensor, classes: Tensor, target_lik: float, condition: Literal['concentrate', 'lower_than', 'greater_than'], concentrate_dim_log_tol: float=1.0, perc_change: float=0.003, return_all: bool=False) -> Tensor:
        """
        Linear modification always happens in the embedding space, as there are
        no guarantees at this point that the reconstruction is faithful.
        """
        org_shape = batch.shape
        assert isinstance(batch, Tensor) and batch.dim() == 2 and batch.shape[0] > 0, 'sample needs to be a non-empty 2D Tensor'
        if self.space_in == 'X':
            batch = self.flow.x_to_b(input=batch, classes=classes)
        clz_int = classes.squeeze().to(dtype=torch.int64, device=batch.device)
        embedded = batch.clone().detach()




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
                t = flow.x_to_e(input=t)
            b, ld = flow.e_to_b(embeddings=t, classes=clz)
            return b, ld
        
        def from_b(b: Tensor, clz: Tensor) -> Tensor:
            t = flow.e_from_b(base=b, classes=clz)[0]
            if not sample_is_embedding:
                t = flow.x_from_e(embeddings=t)
            return t
        
        def lik(t: Tensor, clz: Tensor) -> Tensor:
            return (flow.log_rel_lik_emb(embeddings=t, classes=clz) if sample_is_embedding else flow.log_rel_lik(input=t, classes=clz))
        
        def lik_unagg(t: Tensor, clz: Tensor) -> tuple[Tensor, Tensor]:
            if not sample_is_embedding:
                t = flow.x_to_e(input=t)
            b, b_log_det = flow.e_to_b(embeddings=t, classes=clz)

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
            
            # Modify b according to the condition and target.
            b_prime = torch.vstack(tensors=list(
                normals[clz_int[idx].item()].icdf(Linear._icdf_safe(
                    q=q[idx] + lamb[idx] * u[idx])) for idx in range(b_batch.shape[0])))
            
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