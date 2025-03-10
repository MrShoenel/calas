import torch
from torch import nn, Tensor, device, dtype
from torch.func import jacrev
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
from normflows import ConditionalNormalizingFlow
from normflows.distributions.base import ConditionalDiagGaussian
from normflows.flows import Flow
from typing import Optional, Union, override, Literal, final, Iterator
from .repr import Representation, ReconstructableRepresentation
from ..tools.func import normal_ppf_safe





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

        self._loss_grad_wrt_X = jacrev(func=self.loss_wrt_X)
        self._loss_grad_wrt_E = jacrev(func=self.loss_wrt_E)
        self._loss_wrt_B_grad = jacrev(func=self.loss_wrt_B)

        # TODO: Implement static translation according to conditional means
    

    @property
    def num_dims_X(self) -> int:
        return self.num_dims
    
    @property
    def num_dims_E(self) -> int:
        return self.num_dims
    
    @property
    def num_dims_B(self) -> int:
        return self.num_dims
    

    @override
    def parameters(self, recurse: bool=True) -> Iterator[nn.Parameter]:
        assert recurse, "`recurse` must be set to true in order to get all of the underlying flow's parameters."
        """
        Overridden to always return the parameters of the underlying flow and
        its base distribution. Recurse must be True
        """
        for param in self.flow.parameters(recurse=recurse):
            yield param
    

    @property
    def dev(self) -> device:
        return self._dev_tracker.device
    

    @property
    def dtype(self) -> dtype:
        return self._dev_tracker.dtype
    

    @property
    def can_reconstruct(self) -> bool:
        # TODO: This should perhaps return True for a flow without repr.
        # We shouldn't probably use this property on a flow at all.
        return False
    

    def mean_for_class(self, clazz: int) -> Tensor:
        assert isinstance(clazz, int) and clazz >= 0 and clazz < self.num_classes
        return self.cond_means[clazz]
    

    def log_scale_for_class(self, clazz: int) -> Tensor:
        assert isinstance(clazz, int) and clazz >= 0 and clazz < self.num_classes
        return self.cond_log_scales[clazz]
    

    def ctx_endoder(self, classes: Tensor) -> Tensor:
        clz_int = torch.atleast_1d(classes.squeeze().to(torch.int64))
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
    

    def X_to_E(self, input: Tensor) -> Tensor:
        """
        Forward pass, from data/input space (X) to representation/embedding space (E).
        This flow does not use a representation, so X=E and this function is the identity.
        """
        return input
    

    def X_to_B(self, input: Tensor, classes: Tensor) -> tuple[Tensor, Tensor]:
        """
        Convenience method that performs a full pass from data/input space (X) to base
        space (B), by also passing the X through the representation/embedding space (E)
        first, i.e., X -> E -> B.
        """
        emb = self.X_to_E(input=input)
        return self.E_to_B(embeddings=emb, classes=classes)
    

    def X_from_E(self, embeddings: Tensor) -> Tensor:
        """
        Inverse pass, from representation/embedding space (E) to data/input space (X).
        This flow does not use a representation, so E=X and this function is the identity.
        """
        return embeddings


    def E_to_B(self, embeddings: Tensor, classes: Tensor) -> tuple[Tensor, Tensor]:
        """
        Forward-pass, from data space (X)representation/embedding space (E) to base space (B).
        Returns the result in the base space and the log-determinant.
        """
        return self.flow.inverse_and_log_det(x=embeddings, context=self.ctx_endoder(classes=classes))
    

    def E_from_B(self, base: Tensor, classes: Tensor) -> tuple[Tensor, Tensor]:
        """
        Inverse pass, from base space (B) to representation/embedding space (E).
        Returns the result in the representation/embedding space and the log-determinant.
        """
        return self.flow.forward_and_log_det(z=base, context=self.ctx_endoder(classes=classes))
    

    def X_from_B(self, base: Tensor, classes: Tensor) -> tuple[Tensor, Tensor]:
        """
        Convenience method that performs a full pass from base space (B) all the way back
        to the input/data space (X), by also passing B through the representation/embedding
        space (E) first, i.e., B -> E -> X.
        """
        emb, log_det = self.E_from_B(base=base, classes=classes)
        return self.X_from_E(embeddings=emb), log_det
    

    def log_rel_lik(self, input: Tensor, classes: Tensor) -> Tensor:
        emb = self.X_to_E(input=input)
        return self.log_rel_lik_E(embeddings=emb, classes=classes)
    

    def log_rel_lik_X(self, input: Tensor, classes: Tensor) -> Tensor:
        """Alias of log_rel_lik()."""
        return self.log_rel_lik(input=input, classes=classes)
    

    def log_rel_lik_E(self, embeddings: Tensor, classes: Tensor) -> Tensor:
        return self.flow.log_prob(x=embeddings, context=self.ctx_endoder(classes=classes))
    

    def log_rel_lik_B(self, base: Tensor, classes: Tensor) -> Tensor:
        emb = self.E_from_B(base=base, classes=classes)[0]
        return self.log_rel_lik_E(embeddings=emb, classes=classes)
    

    def loss(self, input: Tensor, classes: Tensor) -> Tensor:
        """
        Computes the forward KL divergence (we estimate the expectation using
        Monte Carlo). In other words, given samples in the data space (X), we
        calculate their average negative log likelihood under the base (B)
        distribution.
        """
        emb = self.X_to_E(input=input)
        return self.loss_wrt_E(embeddings=emb, classes=classes)
    

    def loss_wrt_X(self, input: Tensor, classes: Tensor) -> Tensor:
        """Alias for loss()."""
        return self.loss(input=input, classes=classes)
    

    def loss_wrt_E(self, embeddings: Tensor, classes: Tensor) -> Tensor:
        """
        Computes the forward KL divergence (we estimate the expectation using
        Monte Carlo). In other words, given samples in the representation/
        embeddings space (E), we calculate their average negative log likelihood
        under the base (B) distribution.
        """
        return -torch.mean(input=self.log_rel_lik_E(embeddings=embeddings, classes=classes))
    

    def loss_wrt_B(self, base: Tensor, classes: Tensor) -> Tensor:
        """
        Inverse-transforms B -> E, then calls `loss_E()`.
        """
        emb = self.E_from_B(base=base, classes=classes)[0]
        return self.loss_wrt_E(embeddings=emb, classes=classes)
    

    def loss_wrt_Q(self, q: Tensor, classes: Tensor, loc: Optional[Tensor]=None, scale: Optional[Tensor]=None) -> Tensor:
        """
        Computes the forward KL divergence of quantiles of some B. If location or
        scale are missing, uses this flow's conditional base distribution's. Then,
        the quantiles are used to "reconstitute" b from q, and a loss is calculated
        by calling `loss_B()`.
        """
        clz_int: Tensor = None
        if loc is None or scale is None:
            assert isinstance(classes, Tensor)
            clz_int = torch.atleast_1d(classes.squeeze().to(torch.int64))
        
        if loc is None:
            loc = torch.vstack(tensors=list(
                self.mean_for_class(clazz=clz_int[idx].item()) for idx in range(q.shape[0]))).repeat(1, self.num_dims)
        if scale is None:
            scale = torch.exp(torch.vstack(tensors=list(
                self.log_scale_for_class(clazz=clz_int[idx].item()) for idx in range(b.shape[0]))).repeat(1, self.num_dims))
        
        b: Tensor = torch.vmap(normal_ppf_safe)(q, loc, scale)
        return self.loss_wrt_B(base=b, classes=classes)
    

    def loss_wrt_X_grad(self, inputs: Tensor, classes: Tensor) -> Tensor:
        """
        The derivative of this flow's loss with regard to the inputs (here: original data space).
        This function tells us how the inputs (*not* the flow's parameters) in the original data
        space need to be changed in order to increase/decrease the resulting loss.
        """
        return self._loss_grad_wrt_X(inputs, classes)
    

    def loss_wrt_E_grad(self, embedded: Tensor, classes: Tensor) -> Tensor:
        """
        The derivative of this flow's loss with regard to the inputs (here: the embeddings).
        This function tells us how the inputs (*not* the flow's parameters) in the embedding
        space need to be changed in order to increase/decrease the resulting loss.
        """
        return self._loss_grad_wrt_E(embedded, classes)
    

    def loss_wrt_B_grad(self, base: Tensor, classes: Tensor) -> Tensor:
        """
        The derivative of this flow's loss with regard to base space (B). This function tells
        use how the sample in B space (*not* the flow's parameters) need to be changed in order
        to increase/decrease the resulting loss.
        """
        return self._loss_wrt_B_grad(base, classes)
    

    @final
    def forward(self, *args, **kwargs) -> None:
        raise Exception('Intent not clear, call, e.g., X_to_E, log_prob, loss, ..., etc.')
    

    def sample(self, n_samp: int, classes: Optional[Tensor]=None) -> tuple[Tensor, Tensor, Tensor]:
        """
        Calls `sample_e()` and transforms the sampled embeddings back to the
        original data space (X).
        """
        samp_emb, log_liks, clazzes = self.sample_E(n_samp=n_samp, classes=classes)
        return self.X_from_E(embeddings=samp_emb), log_liks, clazzes
    

    def sample_E(self, n_samp: int, classes: Optional[Tensor]=None) -> tuple[Tensor, Tensor, Tensor]:
        """
        Samples from the flow using optional conditional classes. If no classes
        are given, generates random classes first.
        Note that this method returns samples in the representation/sampling space (E)!

        Returns a tuple of samples, their log probabilities, and the classes of the samples.
        """
        if classes is None:
            classes = torch.randint(low=0, high=self.num_classes, size=(n_samp,)).to(device=self.dev, dtype=torch.int64).squeeze()
        
        return *self.flow.sample(num_samples=n_samp, context=self.ctx_endoder(classes=classes)), classes




class CalasFlowWithRepr(CalasFlow):
    def __init__(self, num_classes: int, flows: list[Flow], repr: Union[Representation, ReconstructableRepresentation], *args, **kwargs):
        super().__init__(num_dims=repr.embed_dim, num_classes=num_classes, flows=flows, *args, **kwargs)
        self.repr = repr
    

    @property
    @override
    def num_dims_X(self) -> int:
        return self.repr.input_dim
    

    @override
    def parameters(self, recurse = True) -> Iterator[nn.Parameter]:
        """
        Always returns the parameters of the underlying flow and its base distribution.
        If `recurse = True`, also yields the parameters of own modules, such as those
        of the representation.
        """
        for param in super().parameters(recurse=True):
            yield param
        
        if recurse:
            for param in self.repr.parameters(recurse=recurse):
                yield param
    

    @property
    @override
    def can_reconstruct(self) -> bool:
        return isinstance(self.repr, ReconstructableRepresentation)
    
    
    @override
    def X_to_E(self, input: Tensor) -> Tensor:
        return self.repr.embed(x=input)
    
    
    @override
    def X_from_E(self, embeddings: Tensor) -> Tensor:
        assert self.can_reconstruct, 'Can only sample if the representation supports reconstruction.'
        return self.repr.reconstruct(embeddings=embeddings)
