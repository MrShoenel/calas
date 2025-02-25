import torch
import numpy as np
import normflows as nf
from torch import nn, Tensor, device
from normflows import ConditionalNormalizingFlow
from normflows.distributions.base import ConditionalDiagGaussian
from normflows.flows import Flow
from typing import Optional






class CalasFlow(nn.Module):
    def __init__(self, num_dims: int, num_classes: int, flows: list[Flow], dev: device=device('cpu'), *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dev = dev
        self.num_classes = num_classes
        self.num_dims = num_dims
        self.base_dist = ConditionalDiagGaussian(shape=num_dims, context_encoder=lambda c: c)
        self.flow = ConditionalNormalizingFlow(q0=self.base_dist, flows=flows)
    

    def ctx_endoder(self, classes: Tensor) -> Tensor:
        clz_int = classes.squeeze().to(torch.int64)
        assert clz_int.dim() == 1
        assert all(c >= 0 and c < self.num_classes for c in clz_int.detach().cpu().tolist())
        extent = self.num_dims * 6. # We'll space normal distributions apart by 6 with an sd of 0.5

        means = torch.tensor(list(-1/2*extent + 3. + (i*6) for i in range(self.num_dims)), device=classes.device)
        log_scales = torch.log(torch.tensor([0.5] * self.num_dims, device=classes.device))

        # We will have to return 2D Tensor, one row for each item's class.
        # In each row, the first N/2 dimensions are the one-hot selected means,
        # and the second N/2 half are the one-hot selected log scales.

        means_1hot = torch.zeros((clz_int.numel(), self.num_dims), device=clz_int.device)
        log_scales_1hot = torch.zeros((clz_int.numel(), self.num_dims), device=clz_int.device)
        for idx in range(clz_int.numel()):
            means_1hot[idx] = means[clz_int[idx]]
            log_scales_1hot[idx] = log_scales[clz_int[idx]]

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
