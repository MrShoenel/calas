import torch
from torch import nn, Tensor, device, dtype
from torch.func import jacrev
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
from normflows import ConditionalNormalizingFlow
from normflows.distributions.base import ConditionalDiagGaussian
from normflows.flows import Flow
from typing import Optional, Union, override, Literal, final
from .repr import Representation, ReconstructableRepresentation





class CalasFlow(nn.Module):
    def __init__(self, num_dims: int, num_classes: int, flows: list[Flow], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_classes = num_classes
        self.num_dims = num_dims
        self.base_dist = ConditionalDiagGaussian(shape=num_dims, context_encoder=lambda c: c)
        self.flow = ConditionalNormalizingFlow(q0=self.base_dist, flows=flows)

        # We keep around this empty buffer because it will be affected when someone
        # calls model.to(..device..). Then, whenever we need to access the device
        # the model is currently on, we can check the device of this buffer.
        self.register_buffer(name='_dev_tracker', tensor=torch.empty(0))

        extent = self.num_classes * 6. # We'll space normal distributions apart by 6 with an sd of 0.5
        self.register_buffer(name='cond_means', tensor=torch.tensor(list(-1/2*extent + i*6. + 3. for i in range(self.num_classes))))
        self.register_buffer(name='cond_log_scales', tensor=torch.log(torch.tensor([0.5]*self.num_classes)))

        self._loss_grad_wrt_inputs = jacrev(func=self.loss)
        self._loss_emb_grad_wrt_inputs = jacrev(func=self.loss_emb)
    

    @property
    def dev(self) -> device:
        return self._dev_tracker.device
    

    @property
    def dtype(self) -> dtype:
        return self._dev_tracker.dtype
    

    @property
    def can_reconstruct(self) -> bool:
        return False
    

    def mean_for_class(self, clazz: int) -> Tensor:
        assert isinstance(clazz, int) and clazz >= 0 and clazz < self.num_classes
        return self.cond_means[clazz]
    

    def log_scale_for_class(self, clazz: int) -> Tensor:
        assert isinstance(clazz, int) and clazz >= 0 and clazz < self.num_classes
        return self.cond_log_scales[clazz]
    

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
            (self.cond_means.repeat(clz_int.numel(), 1) * clz_1hot).sum(dim=1)).T.repeat(1, self.num_dims)
        log_scales_1hot = torch.atleast_2d(
            (self.cond_log_scales.repeat(clz_int.numel(), 1) * clz_1hot).sum(dim=1)).T.repeat(1, self.num_dims)

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
        emb = self.x_to_e(input=input)
        return self.log_rel_lik_emb(embeddings=emb, classes=classes)
    

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
            classes = torch.randint(low=0, high=self.num_classes, size=(n_samp,)).to(device=self.dev, dtype=torch.int64).squeeze()
        
        return *self.flow.sample(num_samples=n_samp, context=self.ctx_endoder(classes=classes)), classes
    

    @final
    def forward(self, *args, **kwargs) -> None:
        raise Exception('Intent not clear, call, e.g., x_to_e, log_prob, loss, ..., etc.')
    

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



class CalasFlowWithRepr(CalasFlow):
    def __init__(self, num_classes: int, flows: list[Flow], repr: Union[Representation, ReconstructableRepresentation], *args, **kwargs):
        super().__init__(num_dims=repr.embed_dim, num_classes=num_classes, flows=flows, *args, **kwargs)

        self.repr = repr
    

    @property
    @override
    def can_reconstruct(self) -> bool:
        return isinstance(self.repr, ReconstructableRepresentation)
    
    
    @override
    def x_to_e(self, input: Tensor) -> Tensor:
        return self.repr.embed(x=input)
    
    
    @override
    def x_from_e(self, embeddings: Tensor) -> Tensor:
        assert self.can_reconstruct, 'Can only sample if the representation supports reconstruction.'
        return self.repr.reconstruct(embeddings=embeddings)


    # def make_random_cod_anomaly(self) -> Tensor:
    #     pass

    # def make_random_cod_anomaly_emb(self) -> Tensor:
    #     pass
    
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
