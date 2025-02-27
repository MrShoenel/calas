import torch
from torch import nn, Tensor
from abc import ABC, abstractmethod
from typing import override


class Representation(nn.Module, ABC):
    def __init__(self, input_dim: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_dim = input_dim
    

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        """
        The (flat) number of dimensions of the representation/embedding. In
        other words, shape[1] of the tensor as produced by `forward()`.
        """
        ... # pragma: no cover


    def forward(self, x: Tensor) -> Tensor:
        """Alias for `embed()`."""
        return self.embed(x=x)
    

    @abstractmethod
    def embed(self, x: Tensor) -> Tensor:
        """
        Takes input from the original data space (X) and embeds it in some custom
        representation that usually has a different number of dimensions.
        """
        ... # pragma: no cover



class ReconstructableRepresentation(Representation):
    @property
    @abstractmethod
    def decoder(self) -> nn.Module:
        """
        An extra module used to decode embeddings. Its output must have the same
        dimensionality as the original input. The decoder is used during
        reconstruction, as well as for computing the loss of this representation.
        """
        ... # pragma: no cover


    def reconstruct(self, embeddings: Tensor) -> Tensor:
        """
        Takes as input embedded data and runs it through the decoder. The shape
        of the given embeddings here is the same as the result from `forward()`
        (or `embed()`).
        """
        result: Tensor = self.decoder(embeddings)
        assert result.dim() == 2 and result.shape[0] == embeddings.shape[0] and result.shape[1] == self.input_dim
        return result


    def loss(self, x: Tensor) -> Tensor:
        prepare = lambda t: nn.functional.softmax(5*nn.functional.tanh(.2*t))

        i = prepare(x)

        i_prime = self.embed(x=x)
        i_prime = self.decoder(i_prime)
        i_prime = prepare(i_prime)

        return nn.functional.kl_div(
            input=torch.log(i), target=i_prime, log_target=False)



class Identity(ReconstructableRepresentation):
    """
    A simple reconstruction that is the identity. It does not modify its inputs
    or outputs in any way. The decoder is the identity, too, such that the
    reconstruction is identical to its embedding.
    """
    def __init__(self, input_dim: int, *args, **kwargs):
        super().__init__(input_dim=input_dim, *args, **kwargs)
        class ID(nn.Module):
            def forward(self, x): x
        self._decoder = ID()
    
    @property
    @override
    def decoder(self) -> nn.Module:
        return self._decoder

    @property
    @override
    def embed_dim(self) -> int:
        return self.input_dim
    
    @override
    def embed(self, x: Tensor) -> Tensor:
        return x
