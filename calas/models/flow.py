import torch
from torch import nn, Tensor, device
from torch.func import vmap, jacrev
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
from normflows import ConditionalNormalizingFlow
from normflows.distributions.base import ConditionalDiagGaussian
from normflows.flows import Flow
from typing import Optional, Union, override, Literal
from .repr import Representation, RepresentationWithReconstruct





class CalasFlow(nn.Module):
    def __init__(self, num_dims: int, num_classes: int, flows: list[Flow], dev: device=device('cpu'), *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dev = dev
        self.num_classes = num_classes
        self.num_dims = num_dims
        self.base_dist = ConditionalDiagGaussian(shape=num_dims, context_encoder=lambda c: c)
        self.flow = ConditionalNormalizingFlow(q0=self.base_dist, flows=flows)

        extent = self.num_classes * 6. # We'll space normal distributions apart by 6 with an sd of 0.5
        self._cond_means = torch.tensor(list(-1/2*extent + 3. + (i*6) for i in range(self.num_classes)), device=self.dev)
        self._cond_log_scales = torch.log(torch.tensor([0.5] * self.num_classes, device=self.dev))
    

    def ctx_endoder(self, classes: Tensor) -> Tensor:
        clz_int = classes.squeeze().to(torch.int64)
        # while torch._C._functorch.is_functorch_wrapped_tensor(clz_int):
        #     clz_int = torch._C._functorch.get_unwrapped(clz_int)
        # assert clz_int.dim() == 1
        # assert all(c >= 0 and c < self.num_classes for c in clz_int.detach().cpu().tolist())
        

        # We will have to return 2D Tensor, one row for each item's class.
        # In each row, the first N/2 dimensions are the one-hot selected means,
        # and the second N/2 half are the one-hot selected log scales.

        clz_1hot = nn.functional.one_hot(clz_int, self.num_classes)

        # means_1hot = (self._cond_means.repeat(clz_int.numel(), 1) * clz_1hot).repeat(1, self.num_dims)
        means_1hot = torch.atleast_2d((self._cond_means.repeat(clz_int.numel(), 1) * clz_1hot).sum(dim=1)).T.repeat(1, self.num_dims)
        # log_scales_1hot = (self._cond_log_scales.repeat(clz_int.numel(), 1) * clz_1hot).repeat(1, self.num_dims)
        log_scales_1hot = torch.atleast_2d((self._cond_log_scales.repeat(clz_int.numel(), 1) * clz_1hot).sum(dim=1)).T.repeat(1, self.num_dims)

        return torch.hstack((means_1hot, log_scales_1hot))
    

    def x_to_b(self, x: Tensor, classes: Tensor) -> tuple[Tensor, Tensor]:
        """This is the forward-pass, from data space (X) to base space (B)."""
        return self.flow.inverse_and_log_det(x=x, context=self.ctx_endoder(classes=classes))
    

    def b_to_x(self, b: Tensor, classes: Tensor) -> tuple[Tensor, Tensor]:
        """This is the inverse pass, from base space (B) to data space (X)."""
        return self.flow.forward_and_log_det(z=b, context=self.ctx_endoder(classes=classes))
    

    def log_prob(self, x: Tensor, classes: Tensor) -> Tensor:
        return self.flow.log_prob(x=x, context=self.ctx_endoder(classes=classes))
    

    def loss(self, x: Tensor, classes: Tensor) -> Tensor:
        """
        Computes the forward KL divergence (we estimate the expectation using
        Monte Carlo). In other words, given samples in the data space (X), we
        calculate their average negative log likelihood under the base (B)
        distribution.
        """
        return -torch.mean(input=self.log_prob(x=x, classes=classes))
    

    def sample(self, n_samp: int, classes: Optional[Tensor]=None) -> tuple[Tensor, Tensor, Tensor]:
        """
        Samples from the flow using optional conditional classes. If no classes
        are given, generates random classes first.

        Returns a tuple of samples, their log probabilities, and the classes of the samples.
        """
        if classes is None:
            classes = torch.randint(low=0, high=self.num_classes, size=(n_samp,), device=self.dev).to(dtype=torch.int64).squeeze()
        
        return *self.flow.sample(num_samples=n_samp, context=self.ctx_endoder(classes=classes)), classes


    def forward(self, *args, **kwargs) -> None:
        raise Exception('Intent not clear, call, e.g., x_to_b, b_to_x, log_prob, or loss.')



class CalasFlowWithRepr(CalasFlow):
    def __init__(self, num_classes: int, flows: list[Flow], repr: Union[Representation, RepresentationWithReconstruct], dev: device=device('cpu'), *args, **kwargs):
        super().__init__(num_dims=repr.embed_dim, num_classes=num_classes, flows=flows, dev=dev, *args, **kwargs)

        self.repr = repr
        self._loss_embedded_grad = jacrev(func=self.loss_embedded)
    

    @property
    def can_reconstruct(self) -> bool:
        return isinstance(self.repr, RepresentationWithReconstruct)
    

    def log_prob_embedded(self, embedded: Tensor, classes: Tensor):
        return self.flow.log_prob(x=embedded, context=self.ctx_endoder(classes=classes))

    @override
    def log_prob(self, x: Tensor, classes: Tensor):
        embedded = self.repr.forward(x=x)
        return self.log_prob_embedded(embedded=embedded, classes=classes)
    

    def embedded_to_b(self, embedded: Tensor, classes: Tensor) -> Tensor:
        return super().x_to_b(x=embedded, classes=classes)

    @override
    def x_to_b(self, x: Tensor, classes: Tensor) -> Tensor:
        embedded = self.repr.forward(x=x)
        return self.embedded_to_b(embedded=embedded, classes=classes)
    

    def b_to_embedded(self, b: Tensor, classes: Tensor):
        return super().b_to_x(b=b, classes=classes)


    @override
    def b_to_x(self, b: Tensor, classes: Tensor):
        assert self.can_reconstruct, 'Can only sample if the representation supports reconstruction.'

        embedded = self.b_to_embedded(b=b, classes=classes)
        reconst = self.repr.reconstruct(embeddings=embedded)
        return reconst
    

    def sample_embedded(self, n_samp: int, classes: Optional[Tensor]=None) -> tuple[Tensor, Tensor, Tensor]:
        return super().sample(n_samp=n_samp, classes=classes)
    

    @override
    def sample(self, n_samp: int, classes: Optional[Tensor]=None) -> tuple[Tensor, Tensor, Tensor]:
        assert self.can_reconstruct, 'Can only sample if the representation supports reconstruction.'
        
        samples, log_probs, clz = self.sample_embedded(n_samp=n_samp, classes=classes)
        reconst = self.repr.reconstruct(embeddings=samples)
        return reconst, log_probs, clz
    

    def loss_embedded(self, embedded: Tensor, classes: Tensor) -> Tensor:
        return -torch.mean(input=self.log_prob_embedded(embedded=embedded, classes=classes))
    

    def loss_embedded_grad(self, embedded: Tensor, classes: Tensor) -> Tensor:
        """
        The derivative of this flow's loss with regard to the inputs (here: the embeddings).
        This function tells us how the inputs (*not* the flow's parameters) in the embedding
        space need to be changed in order to increase/decrease the resulting loss.
        """
        return self._loss_embedded_grad(embedded, classes)
    

    @override
    def loss(self, x: Tensor, classes: Tensor) -> Tensor:
        embedded = self.repr.forward(x=x)
        return self.loss_embedded(embedded=embedded, classes=classes)


    def deriv_wrt_input_embeddings(self, embeddings: Tensor, classes: Tensor) -> Tensor:
        assert embeddings.dim() == 2 and embeddings.shape[1] == self.repr.embed_dim # to avoid passing raw input here
        assert embeddings.requires_grad

        was_training = self.training
        self.eval()

        result = self.loss_embedded_grad(embedded=embeddings, classes=classes)
        if was_training:
            self.train()
        return result

    
    def make_CoD_batch_random_embedding(self, nominal: Tensor, classes: Tensor, distr: Literal['Normal', 'Uniform']|None=None, num_dims: tuple[int,int]=(0,1), mode: Literal['replace', 'add']|None=None, use_grad_dir: bool|None=None, normalize: bool|None=True, verbose: bool=False) -> Tensor:
        embedded = self.repr.forward(x=nominal).detach_()
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
            grad = self.deriv_wrt_input_embeddings(embeddings=embedded, classes=classes)
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
