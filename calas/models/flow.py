import torch
from torch import nn, Tensor, device
from torch.func import vmap, jacrev
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
from normflows import ConditionalNormalizingFlow
from normflows.distributions.base import ConditionalDiagGaussian
from normflows.flows import Flow
from typing import Optional, Union, override, Literal, final
from .repr import Representation, RepresentationWithReconstruct





class CalasFlow(nn.Module):
    def __init__(self, num_dims: int, num_classes: int, flows: list[Flow], dev: device=device('cpu'), *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dev = dev
        self.num_classes = num_classes
        self.num_dims = num_dims
        self.base_dist = ConditionalDiagGaussian(shape=num_dims, context_encoder=lambda c: c)
        self.flow = ConditionalNormalizingFlow(q0=self.base_dist, flows=flows)

        self._cond_means = torch.tensor(list(self.mean_for_class(clazz=i) for i in range(self.num_classes)), device=self.dev)
        self._cond_log_scales = torch.log(torch.tensor(list(self.scale_for_class(clazz=i) for i in range(self.num_classes)), device=self.dev))
    

    def mean_for_class(self, clazz: int) -> float:
        assert isinstance(clazz, int) and clazz >= 0 and clazz < self.num_classes
        extent = self.num_classes * 6. # We'll space normal distributions apart by 6 with an sd of 0.5
        return -1/2*extent + clazz*6. + 3.
    

    def scale_for_class(self, clazz: int) -> float:
        return 0.5 # Currently, we will use the exact same for each and every feature.
    

    def ctx_endoder(self, classes: Tensor) -> Tensor:
        clz_int = classes.squeeze().to(torch.int64)
        if not torch._C._functorch.is_functorch_wrapped_tensor(clz_int):
            assert clz_int.dim() == 1
            assert all(c >= 0 and c < self.num_classes for c in clz_int.detach().cpu().tolist())

        # We will have to return 2D Tensor, one row for each item's class.
        # In each row, the first N/2 dimensions are the one-hot selected means,
        # and the second N/2 half are the one-hot selected log scales.

        clz_1hot = nn.functional.one_hot(clz_int, self.num_classes)
        means_1hot = torch.atleast_2d(
            (self._cond_means.repeat(clz_int.numel(), 1) * clz_1hot).sum(dim=1)).T.repeat(1, self.num_dims)
        log_scales_1hot = torch.atleast_2d(
            (self._cond_log_scales.repeat(clz_int.numel(), 1) * clz_1hot).sum(dim=1)).T.repeat(1, self.num_dims)

        return torch.hstack((means_1hot, log_scales_1hot))
    

    def x_to_e(self, input: Tensor) -> Tensor:
        """
        Forward pass, from data/input space (X) to representation/embedding space (E).
        This flow does not use a representation, so X=E and this function is the identity.
        """
        return input
    

    def x_from_e(self, embeddings: Tensor) -> Tensor:
        """
        Inverse pass, from representation/embedding space (E) to data/input space (X).
        This flow does not use a representation, so E=X and this function is the identity.
        """
        return embeddings


    def e_to_b(self, embeddings: Tensor, classes: Tensor) -> tuple[Tensor, Tensor]:
        """
        Forward-pass, from data space (X)representation/embedding space (E) to base space (B).
        Returns the result in the base space and the log-determinant.
        """
        return self.flow.inverse_and_log_det(x=embeddings, context=self.ctx_endoder(classes=classes))
    

    def e_from_b(self, base: Tensor, classes: Tensor) -> tuple[Tensor, Tensor]:
        """
        Inverse pass, from base space (B) to representation/embedding space (E).
        Returns the result in the representation/embedding space and the log-determinant.
        """
        return self.flow.forward_and_log_det(z=base, context=self.ctx_endoder(classes=classes))
    

    def log_rel_lik(self, input: Tensor, classes: Tensor) -> Tensor:
        return self.log_rel_lik_emb(embeddings=self.x_to_e(input=input), classes=classes)
    

    def log_rel_lik_emb(self, embeddings: Tensor, classes: Tensor) -> Tensor:
        return self.flow.log_prob(x=embeddings, context=self.ctx_endoder(classes=classes))
    

    def loss(self, input: Tensor, classes: Tensor) -> Tensor:
        """
        Computes the forward KL divergence (we estimate the expectation using
        Monte Carlo). In other words, given samples in the data space (X), we
        calculate their average negative log likelihood under the base (B)
        distribution.
        """
        emb = self.x_to_e(input=input)
        return self.loss_emb(embeddings=emb, classes=classes)
    

    def loss_emb(self, embeddings: Tensor, classes: Tensor) -> Tensor:
        """
        Computes the forward KL divergence (we estimate the expectation using
        Monte Carlo). In other words, given samples in the representation/
        embeddings space (E), we calculate their average negative log likelihood
        under the base (B) distribution.
        """
        return -torch.mean(input=self.log_rel_lik_emb(embeddings=embeddings, classes=classes))
    

    def sample(self, n_samp: int, classes: Optional[Tensor]=None) -> tuple[Tensor, Tensor, Tensor]:
        """
        Calls `sample_e()` and transforms the sampled embeddings back to the
        original data space (X).
        """
        samp_emb, log_liks, clazzes = self.sample_emb(n_samp=n_samp, classes=classes)
        return self.x_from_e(embeddings=samp_emb), log_liks, clazzes
    

    def sample_emb(self, n_samp: int, classes: Optional[Tensor]=None) -> tuple[Tensor, Tensor, Tensor]:
        """
        Samples from the flow using optional conditional classes. If no classes
        are given, generates random classes first.
        Note that this method returns samples in the representation/sampling space (E)!

        Returns a tuple of samples, their log probabilities, and the classes of the samples.
        """
        if classes is None:
            classes = torch.randint(low=0, high=self.num_classes, size=(n_samp,), device=self.dev).to(dtype=torch.int64).squeeze()
        
        return *self.flow.sample(num_samples=n_samp, context=self.ctx_endoder(classes=classes)), classes
    

    @final
    def forward(self, *args, **kwargs) -> None:
        raise Exception('Intent not clear, call, e.g., x_to_e, log_prob, loss, ..., etc.')
    

    def make_linear_global_anomaly(self, input: Tensor, classes: Tensor, likelihood: Literal['increase', 'decrease']) -> Tensor:
        """
        Change the likelihood of each sample in data under the current model.
        Each sample is linearly modified until its likelihood is above or below
        the maximum/minimum likelihood of the batch.
        """
        train_before = self.training
        self.eval()

        clz_int = classes.squeeze().to(torch.int64).to(device=input.device)
        emb = self.x_to_e(input=input)
        emb_lik = self.log_rel_lik_emb(embeddings=emb, classes=classes)
        thresh = emb_lik.max().item() if likelihood == 'increase' else emb_lik.min().item()
        
        from torch.distributions.normal import Normal
        normals = { clz: Normal(
            loc=torch.full(size=(self.num_dims,), fill_value=self.mean_for_class(clazz=clz)),
            scale=torch.full(size=(self.num_dims,), fill_value=self.scale_for_class(clazz=clz))) for clz in range(self.num_classes) }
        
        def icdf_save(q: Tensor, tol: float=1e-7) -> Tensor:
            return torch.clip(input=q, min=tol, max=1.-tol)


        results: list[Tensor] = [] # Will hold all samples that work.
        steps = 0

        with torch.no_grad():
            emb = self.x_to_e(input=input.clone()).detach()

            while steps < 20 and len(results) < emb.shape[0]:
                steps += 1
                b_batch = self.e_to_b(embeddings=emb, classes=clz_int)[0].detach()
                u = 1e-5 + torch.rand_like(b_batch) * .003 # ~(0, 0.3]% change

                q = torch.vstack(tensors=list(
                    normals[clz_int[idx].item()].cdf(b_batch[idx]) for idx in range(b_batch.shape[0])))
                if likelihood == 'decrease':
                    # Do not try to make any samples worse that are already beyond good!
                    mask = torch.where((q.min(dim=1).values < 1e-7) | (q.max(dim=1).values > 1-1e-7), True, False)
                    emb = emb[~mask]
                    clz_int = clz_int[~mask]
                    continue
                
                lamb = torch.where(q < 0.5, -1., 1.) * (1. if likelihood == 'decrease' else -1.)
                b_prime = torch.vstack(tensors=list(
                    normals[clz_int[idx].item()].icdf(icdf_save(q=q[idx] + lamb[idx] * u[idx])) for idx in range(b_batch.shape[0])))
                
                # Let's forward the modified b (b_prime) and check the resulting likelihoods!
                emb_prime = self.e_from_b(base=b_prime, classes=clz_int)[0]
                emb_prime_liks = self.log_rel_lik_emb(embeddings=emb_prime, classes=clz_int)

                mask = torch.where(emb_prime_liks < thresh, True, False) if likelihood == 'decrease' else\
                    torch.where(emb_prime_liks > thresh, True, False)
                
                done = emb_prime[mask]
                if done.shape[0] > 0:
                    results.append(self.x_from_e(embeddings=done))
                clz_int = clz_int[~mask]
                emb = emb_prime[~mask]

        if train_before:
            self.train()
        if len(results) == 0:
            return torch.empty((0, emb.shape[1]))
        return torch.vstack(tensors=results)



class CalasFlowWithRepr(CalasFlow):
    def __init__(self, num_classes: int, flows: list[Flow], repr: Union[Representation, RepresentationWithReconstruct], dev: device=device('cpu'), *args, **kwargs):
        super().__init__(num_dims=repr.embed_dim, num_classes=num_classes, flows=flows, dev=dev, *args, **kwargs)

        self.repr = repr
        self._loss_grad_wrt_inputs = jacrev(func=self.loss)
        self._loss_emb_grad_wrt_inputs = jacrev(func=self.loss_emb)
    

    @property
    def can_reconstruct(self) -> bool:
        return isinstance(self.repr, RepresentationWithReconstruct)

    @override
    def log_rel_lik(self, x: Tensor, classes: Tensor) -> Tensor:
        embedded = self.repr.forward(x=x)
        return self.log_rel_lik_emb(embeddings=embedded, classes=classes)
    
    @override
    def x_to_e(self, input: Tensor) -> Tensor:
        return self.repr.embed(x=input)
    
    @override
    def x_from_e(self, embeddings: Tensor) -> Tensor:
        assert self.can_reconstruct, 'Can only sample if the representation supports reconstruction.'
        return self.repr.reconstruct(embeddings=embeddings)
    

    def loss_grad_wrt_input(self, embedded: Tensor, classes: Tensor) -> Tensor:
        """
        The derivative of this flow's loss with regard to the inputs (here: the embeddings).
        This function tells us how the inputs (*not* the flow's parameters) in the embedding
        space need to be changed in order to increase/decrease the resulting loss.
        """
        return self._loss_grad_wrt_inputs(embedded, classes)
    

    def loss_emb_grad_wrt_input(self, embedded: Tensor, classes: Tensor) -> Tensor:
        """
        The derivative of this flow's loss with regard to the inputs (here: the embeddings).
        This function tells us how the inputs (*not* the flow's parameters) in the embedding
        space need to be changed in order to increase/decrease the resulting loss.
        """
        return self._loss_emb_grad_wrt_inputs(embedded, classes)


    def make_random_cod_anomaly(self) -> Tensor:
        pass

    def make_random_cod_anomaly_emb(self) -> Tensor:
        pass
    
    def make_CoD_batch_random_emb(self, nominal: Tensor, classes: Tensor, distr: Literal['Normal', 'Uniform']|None=None, num_dims: tuple[int,int]=(0,1), mode: Literal['replace', 'add']|None=None, use_grad_dir: bool|None=None, normalize: bool|None=True, verbose: bool=False) -> Tensor:
        embedded = self.x_to_e(input=nominal).detach_()
        # Note that the norm is for each sample!
        mean, std, norm = embedded.mean(dim=0), embedded.std(dim=0), torch.atleast_2d(embedded.norm(dim=1, p=1)).T
        
        if distr is None:
            distr = ['Normal', 'Uniform'][torch.randint(low=0, high=2, size=(1,)).item()]
        if verbose:
            print(f'distr={distr}')
        
        dist = Normal(loc=mean, scale=std, validate_args=True) if distr == 'Normal' else Uniform(low=embedded.min(dim=0).values, high=embedded.max(dim=0).values)
        cod = dist.sample((nominal.shape[0],))

        if use_grad_dir is None:
            use_grad_dir = [True, False][torch.randint(low=0, high=2, size=(1,)).item()]
        if use_grad_dir:
            embedded.requires_grad = True
            was_training = self.training
            self.eval()
            grad = self.loss_emb_grad_wrt_input(embedded=embedded, classes=classes)
            if was_training:
                self.train()
            cod = (torch.sign(grad) * torch.abs(cod)).detach()

        if mode is None:
            mode = ['replace', 'add'][torch.randint(low=0, high=2, size=(1,)).item()]
        if verbose:
            print(f'mode={mode}')
        
        assert isinstance(num_dims, tuple)
        a, b = num_dims
        assert isinstance(a, int) and isinstance(b, int) and a <= b
        use_num_dims = a if a == b else torch.randint(low=a, high=b+1, size=(1,)).item()
        
        if verbose:
            print(f'use_num_dims={use_num_dims}')
        
        
        indices = torch.randperm(embedded.shape[1])[0:use_num_dims]
        temp = embedded.clone().detach()

        if mode == 'replace':
            temp[:, indices] = cod[:, indices]
            cod = temp
        elif mode == 'add':
            temp[:, indices] = 0.75 * temp[:, indices] + 0.25 * cod[:, indices]
            cod = temp

        if normalize is None:
            normalize = [True, False][torch.randint(low=0, high=2, size=(1,)).item()]
        if normalize:
            cod_norm = torch.atleast_2d(cod.norm(dim=1, p=1)).T
            cod.div_(cod_norm).mul_(norm)
        return cod
