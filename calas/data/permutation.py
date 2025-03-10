import torch
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from torch.func import jacrev
from ..models.flow import CalasFlow
from ..tools.func import normal_cdf, normal_ppf_safe, normal_log_pdf
from ..tools.mixin import NoGradNoTrainMixin
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
from typing import Generic, TypeVar, Optional, Literal, override, final, Callable, Union
from abc import ABC, abstractmethod
from enum import Enum, StrEnum
from warnings import warn
from dataclasses import dataclass


T = TypeVar(name='T', bound=CalasFlow)


class SampleTooSmallException(Exception): pass


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



@final
class GradientScaling(Enum):
    NoScale = 1
    Normalize = 2
    NormalizeInv = 3
    Softmax = 4
    SoftmaxInv = 5


@dataclass
class Gradient:
    abs: Tensor
    sign: Tensor
    weight: Union[Tensor, float]
    scale: GradientScaling
    likelihood: Likelihood

    @property
    def sign_weight(self) -> tuple[Tensor, Tensor]:
        return self.sign, self.weight

    @property
    def abs_sign_weight(self) -> tuple[Tensor, Tensor, Union[Tensor, float]]:
        return self.abs, self.sign, self.weight

    @staticmethod
    def prepare(grad: Tensor, likelihood: Likelihood, scale: GradientScaling=GradientScaling.SoftmaxInv) -> 'Gradient':
        g_sign = torch.sign(grad) * (-1. if likelihood == Likelihood.Increase else 1.)
        g_abs = torch.abs(grad)
        del grad
        
        g_weight: Union[Tensor, float] = None
        match scale:
            case GradientScaling.NoScale:
                g_weight = 1.0
            case GradientScaling.Normalize:
                g_weight = g_abs / g_abs.max()
            case GradientScaling.NormalizeInv:
                recp = g_abs.reciprocal()
                g_weight = recp / recp.max()
            case GradientScaling.Softmax:
                g_weight = F.softmax(g_abs, dim=1)
            case GradientScaling.SoftmaxInv:
                g_weight = F.softmax(g_abs.reciprocal(), dim=1)
        
        return Gradient(abs=g_abs, sign=g_sign, weight=g_weight, scale=scale, likelihood=likelihood)


class GradientMixin():
    def __init__(self, scaling: GradientScaling):
        self.grad_scaling = scaling

    def prepare_grad(self, grad: Tensor, likelihood: Likelihood) -> Gradient:
        return Gradient.prepare(grad=grad, likelihood=likelihood, scale=self.grad_scaling)


class Permute(NoGradNoTrainMixin[T], Generic[T], ABC):
    """
    This is the base class for all permutations that can be applied to a sample
    in order to change it, such that it incurs a lower or higher loss.
    """

    def __init__(self, flow: T, u_min: float=0.01, u_max: float=0.01, u_frac_negative: float=0.0, seed: Optional[int]=0):
        NoGradNoTrainMixin.__init__(self=self, module=flow)
        self.flow = flow
        self.u_min = u_min
        self.u_max = u_max
        self.u_frac_negative = u_frac_negative
        self.gen = np.random.default_rng(seed=seed)
    

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


class Data2Data_Grad(PermuteData[T], GradientMixin):
    """
    A same-space permutation that updates the input using the gradient with
    regard to the appropriate loss function.
    """
    def __init__(self, flow: T, space: Space, u_min: float=0.01, u_max: float=0.01, grad_scale: GradientScaling=GradientScaling.NormalizeInv, seed: Optional[int]=0):
        PermuteData.__init__(self=self, flow=flow, seed=seed, u_min=u_min, u_max=u_max)
        GradientMixin.__init__(self=self, scaling=grad_scale)

        if space == Space.Quantiles:
            raise Exception(f'The Quantiles-space is not supported. Modifying data through their quantiles requires the assumption of a (normal) distribution. That means you are effectively doing a normal-2-normal (distributional) permutation using gradient information. That is already covered in classes `Normal2Normal_Grad` (using one of the methods `loc_scale`, `quantiles`, or `hybrid`) and `Normal2Normal_NoGrad` (using method `quantiles`).')
        
        self.space = space
    
    
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
        """
        Modifies a sample by taking the derivative with respect to it. This is a
        distribution-free attempt.
        """

        if likelihood == Likelihood.Randomize:
            raise Exception('This permutation explicitly leverages gradient information. If you require a random permutation, try another class.')
        
        clz_int = torch.atleast_1d(classes.squeeze().to(torch.int64))

        # 'batch' could be in any of these 3 spaces.
        loss_fn_grad = self.flow.loss_wrt_X_grad if self.space == Space.Data else (self.flow.loss_wrt_E_grad if self.space == Space.Embedded else self.flow.loss_wrt_B_grad)

        # The gradient points in the direction where the *loss* gets larger,
        # which is the same as the likelihood getting *lower*. So, we have to
        # conditionally inverse their relation as per default, it will decrease
        # the likelihood. Gradient.prepare takes care of that by inversing the
        # sign when likelihood=Increase.
        grad = loss_fn_grad(batch, clz_int)
        grad_sign, grad_weight = self.prepare_grad(grad=grad, likelihood=likelihood).sign_weight
        
        u = self.u_like(input=batch)
        batch_prime = batch + grad_sign * grad_weight * u
        return batch_prime


class Normal2Normal_Grad(Dist2Dist[T], GradientMixin):
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
    :code:`Normal2Normal_NoGrad`, which might give results as good (and sometimes
    better) compared to this class.

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
    def __init__(self, flow: T, method: Literal['loc_scale', 'quantiles', 'hybrid'], u_min: float=0.001, u_max: float=0.05, u_frac_negative: float=0.0, grad_scaling: GradientScaling=GradientScaling.Softmax, seed: Optional[int]=0):
        Dist2Dist.__init__(self=self, flow=flow, seed=seed, u_min=u_min, u_max=u_max, u_frac_negative=u_frac_negative)
        GradientMixin.__init__(self=self, scaling=grad_scaling)

        self.method = method
        self.u_min = u_min
        self.u_max = u_max
    

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

        use_means, use_stds = self.averaged_locs_and_scales(batch=batch, classes=classes)
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

        means_grad_sign, means_grad_weight = self.prepare_grad(grad=means_grad, likelihood=likelihood).sign_weight
        stds_grad_sign, stds_grad_weight = self.prepare_grad(grad=stds_grad, likelihood=likelihood).sign_weight

        means_prime = use_means + means_grad_sign * means_grad_weight * u
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

        grad_sign, grad_weight = self.prepare_grad(grad=grad, likelihood=likelihood).sign_weight
        
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


class Normal2Normal_NoGrad(Dist2Dist[T], GradientMixin):
    """
    Performs distributional modifications similar to :code:`Normal2Normal_Grad`, but without
    using gradient information of the underlying entire model (e.g., the entire flow and its
    representation). It can, however, use very light gradient information of the base distribution,
    which is computationally inexpensive.

    NOTE: This method, when used without the (base) gradient, become very sensitive to the choice
    of `u`. As a rule of thumb, the more well-trained the model becomes, the smaller `u` should be.
    Otherwise, the model will likely overshoot and not produce the desired likelihoods. Since it is
    considered computationally cheap, it is recommended to set `use_loc_scale_base_grad=True` (this
    is the default) and use some some gradient scaling, as this will effectively allow to use a
    constant `u`.

    NOTE: This class operates in base space (B) exclusively and modifies samples there. Samples are
    modified using one of two (three) methods:

    The first method, `loc_scale`, modifies a sample under its assumed normal distribution by first
    computing its averaged means and stds and then modifying those to increase or decrease the
    likelihood of the sample under a normal distribution with these parameters.
    This method has an extra flag, `use_loc_scale_base_grad`. If `False`, means and scales are
    altered step-wise. The means are increased or decreased using a fraction of the standard
    deviation in each call. The scales are multiplied with a constant factor, which is given to the
    constructor: `stds_step_perc`.
    When `use_loc_scale_base_grad=True`, then we take the gradient of the sample under the assumed
    normal distribution w.r.t. the given means and scales. Then, the means and scales are altered
    using that information. Even though we use the gradient here, this method is not considered to
    be expensive, because it is just the gradient of the normal distribution's PDF, not the entire
    flow and its representation.

    The second method, `quantiles`, is the first, initially suggested method for adjusting the
    likelihood of samples under a normalizing flow with a normal base distribution. It *linearly*
    modifies the sample by altering its quantiles under an assumed and averaged normal distribution.
    For increasing the likelihood, it alters the quantiles such that they move towards the median.
    For decreasing the likelihood, the quantiles are pushed away from the median.
    """
    def __init__(self, flow: T, method: Literal['loc_scale', 'quantiles'], use_loc_scale_base_grad: bool=True, u_min: float=0.001, u_max: float=0.001, u_frac_negative: float=0.0, locs_scales_mode: LocsScales=LocsScales.Individual, locs_scales_flow_frac: float=0.2, stds_step_perc: float=0.025, grad_scaling: GradientScaling=GradientScaling.Normalize, seed: Optional[int]=0):
        Dist2Dist.__init__(self=self, flow=flow, seed=seed, u_min=u_min, u_max=u_max, u_frac_negative=u_frac_negative, locs_scales_mode=locs_scales_mode, locs_scales_flow_frac=locs_scales_flow_frac)
        GradientMixin.__init__(self=self, scaling=grad_scaling)

        self.stds_step_perc = stds_step_perc
        self.method = method
        self.use_loc_scale_base_grad = use_loc_scale_base_grad

        if self.use_loc_scale_base_grad:
            self.lik_fn_grad = jacrev(func=lambda x, loc, scale: normal_log_pdf(x=x, loc=loc, scale=scale).sum(dim=1).mean(), argnums=(1,2))
    

    @property
    @override
    def space_in(self) -> Space:
        return Space.Base
    

    @property
    @override
    def space_out(self) -> Space:
        return Space.Base
    

    def permute_loc_scale_step(self, use_means: Tensor, use_stds: Tensor, means_sign: Tensor, u: Tensor, likelihood: Likelihood) -> Tensor:
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

        return means_prime, stds_prime
    

    def permute_loc_scale_base_grad(self, batch: Tensor, use_means: Tensor, use_stds: Tensor, likelihood: Likelihood, u: Tensor) -> Tensor:
        means_grad, stds_grad = self.lik_fn_grad(batch, use_means, use_stds)

        means_grad_sign, means_grad_weight = self.prepare_grad(grad=means_grad, likelihood=likelihood).sign_weight
        stds_grad_sign, stds_grad_weight = self.prepare_grad(grad=stds_grad, likelihood=likelihood).sign_weight

        means_prime = use_means + means_grad_sign * means_grad_weight * u
        stds_prime = use_stds + stds_grad_sign * stds_grad_weight * u

        return means_prime, stds_prime
    

    def permute_quantiles(self, q: Tensor, u: Tensor, use_means: Tensor, use_stds: Tensor, likelihood: Likelihood) -> Tensor:
        """
        This is the first, originally proposed method. It *linearly* (depending on `u`) modifies the
        sample by altering its quantiles under the currently assumed normal distribution. Usually,
        `u` is drawn uniformly (hence the name), which means that the modified sample will approx.
        follow the same distribution(-family; here: normal).
        """
        lamb = torch.where(q < 0.5, 1., -1.).to(dtype=q.dtype)
        if likelihood == Likelihood.Decrease:
            lamb *= -1.
        
        q_prime = q + lamb * u
        b_prime = torch.vmap(normal_ppf_safe)(q_prime, use_means, use_stds)
        return b_prime


    def permute(self, batch: Tensor, classes: Tensor, likelihood: Likelihood) -> Tensor:
        """
        Approximate implementation of old  method of class in line 461.

        NOTE: We don't actually verify that `batch` is in the correct space
        except for testing its dimensionality.
        """
        clz_int = torch.atleast_1d(classes.squeeze().to(torch.int64))
        assert batch.shape[1] == self.flow.num_dims

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
        
        # Here in this method, if we wanted to randomize the resulting likelihoods,
        # we would control it by setting up `u` to be exactly 1.0 with a frac of 0.5.
        # We could also choose 
        u = self.u_like(means_sign)
        q = torch.vmap(normal_cdf)(batch, use_means, use_stds)

        batch_prime: Tensor = None
        if self.method == 'quantiles':
            batch_prime = self.permute_quantiles(q=q, u=u, use_means=use_means, use_stds=use_stds, likelihood=likelihood)
        else:
            means_prime: Tensor = None
            stds_prime: Tensor = None
        
            # else (not quantiles):
            if self.use_loc_scale_base_grad:
                means_prime, stds_prime = self.permute_loc_scale_base_grad(batch=batch, use_means=use_means, use_stds=use_stds, likelihood=likelihood, u=u)
            else:
                means_prime, stds_prime = self.permute_loc_scale_step(use_means=use_means, use_stds=use_stds, means_sign=means_sign, u=u, likelihood=likelihood)
            
            batch_prime = torch.vmap(normal_ppf_safe)(q, means_prime, stds_prime)
        
        return batch_prime


class CurseOfDimDataPermute(PermuteData[T], GradientMixin):
    """
    Permutations applied directly to the data that exploit the curse of
    dimensionality in order to systematically and gradully destroy their
    structure.

    NOTE: This is a batch-2-batch permutation, that is, two or more samples are
    required as input in order to compute some meaningful permutations.
    NOTE: This permutation works with arbitrary spaces, as long as you do not
    use the gradient direction. For some arbitrary space, set this to `False`
    and specify an arbitrary :code:`Space`.
    """
    def __init__(self, flow: T, space: Space, distr: Literal['Normal', 'Uniform']|None=None, num_dims: tuple[int,int]=(1,1), mode: Literal['replace', 'add']|None=None, use_grad_dir: bool|None=None, normalize: bool|None=True, grad_scaling: GradientScaling=GradientScaling.NoScale, u_min: float=0.01, u_max: float=0.01, u_frac_negative: float=0.0, seed: Optional[int]=0):
        PermuteData.__init__(self=self, flow=flow, seed=seed, u_min=u_min, u_max=u_max, u_frac_negative=u_frac_negative)
        GradientMixin.__init__(self=self, scaling=grad_scaling)
        
        if space == Space.Quantiles and use_grad_dir:
            raise Exception('Taking the loss w.r.t. quantiles requires the assumption of a distribution, which we do not have here. Choose another space or disable `use_grad_dir`.')

        self.space = space
        self.distr = distr
        self.num_dims = num_dims
        self.mode = mode
        self.use_grad_dir = use_grad_dir
        self.normalize = normalize
    

    @property
    @override
    def space_in(self) -> Space:
        return self.space
    
    @property
    @override
    def space_out(self) -> Space:
        return self.space
    
    def permute(self, batch: Tensor, classes: Tensor, likelihood: Optional[Likelihood]=None) -> Tensor:
        """
        NOTE: The argument `likelihood` is not supported here and ignored.
        """

        assert batch.dim() == 2
        if batch.shape[0] < 2:
            raise SampleTooSmallException('A 2D tensor with two or more samples is required.')

        mean, std, norm = batch.mean(dim=0), batch.std(dim=0), torch.atleast_2d(batch.norm(dim=1, p=1)).T

        distr = self.distr
        if distr is None:
            distr = ['Normal', 'Uniform'][self.gen.integers(low=0, high=2, size=(1,)).item()]
        
        dist = Normal(loc=mean, scale=std, validate_args=True) if distr == 'Normal' else Uniform(low=batch.min(dim=0).values, high=batch.max(dim=0).values)
        cod = dist.sample((batch.shape[0],))

        use_grad_dir = self.use_grad_dir
        if use_grad_dir is None:
            use_grad_dir = [True, False][self.gen.integers(low=0, high=2, size=(1,)).item()]
        if use_grad_dir:
            grad_fn = self.flow.loss_wrt_X_grad if self.space == Space.Data else (self.flow.loss_wrt_E_grad if self.space == Space.Embedded else self.flow.loss_wrt_B_grad)
            
            grad = grad_fn(batch, classes)
            grad_sign, grad_weight = self.prepare_grad(grad=grad, likelihood=likelihood).sign_weight

            u = self.u_like(cod)
            cod = cod + grad_sign * grad_weight * u
        
        num_dims = self.num_dims
        assert isinstance(num_dims, tuple)
        a, b = num_dims
        assert isinstance(a, int) and isinstance(b, int) and a > 0 and a <= b
        use_num_dims = a if a == b else self.gen.integers(low=a, high=b+1, size=(1,)).item()
        
        
        indices = self.gen.permutation(batch.shape[1]).tolist()[0:use_num_dims]
        temp = batch.clone()

        mode = self.mode
        if mode is None:
            mode = ['replace', 'add'][self.gen.integers(low=0, high=2, size=(1,)).item()]

        if mode == 'replace':
            temp[:, indices] = cod[:, indices]
            cod = temp
        elif mode == 'add':
            temp[:, indices] = 0.5 * temp[:, indices] + 0.5 * cod[:, indices]
            cod = temp

        normalize = self.normalize
        if normalize is None:
            normalize = [True, False][self.gen.integers(low=0, high=2, size=(1,)).item()]
        if normalize:
            cod_norm = torch.atleast_2d(cod.norm(dim=1, p=1)).T
            cod.div_(cod_norm).mul_(norm)
        
        return cod


class PermuteDims(PermuteData[T]):
    """
    Permutes samples by randomly swapping their dimensions. Can work in any
    space. It is not possible the indicate a desired likelihood, either.
    """

    def __init__(self, flow: T, space: Space, num_dims: tuple[int, int]=(2,4), change_prob: float=0.5, seed: Optional[int]=0):
        super().__init__(flow=flow, seed=seed)
        if space == Space.Embedded or space == Space.Base:
            warn('Permuting dimensions in the Embedding (E) or Base (B) space will not change the likelihood of samples under a normalizing flow!')
        assert isinstance(num_dims, tuple) and num_dims[0] <= num_dims[1] and (num_dims[0] % 2) == 0 and (num_dims[1] % 2) == 0
        self.space = space
        self.num_dims = num_dims
        self.change_prob = change_prob
    
    @property
    @override
    def space_in(self) -> Space:
        return self.space
    
    @property
    @override
    def space_out(self) -> Space:
        return self.space
    
    def permute(self, batch: Tensor, classes: Optional[Tensor]=None, likelihood: Optional[Likelihood]=None) -> Tensor:
        """
        NOTE: The arguments `classes` and `likelihood` are completely ignored.
        """
        a, b = self.num_dims
        assert a >= 0 and b <= batch.shape[1]
        use_num_dims: int = 0
        if a == b:
            use_num_dims = a
        else:
            temp = np.array(list(range(a, b+1, 2)))
            use_num_dims = self.gen.choice(a=temp, size=1).item()
        
        if use_num_dims == 0 or self.change_prob < 1e-15:
            return batch
        if self.change_prob > 0:
            if not self.gen.choice(a=[True, False], p=[self.change_prob, 1.0 - self.change_prob], size=1, replace=False).item():
                return batch # we chose not to change
        
        perm = self.gen.permutation(batch.shape[1])[0:(use_num_dims)].tolist()
        new_idx = list(range(batch.shape[1]))

        for i in range(0, use_num_dims, 2):
            p1 = perm[i]
            p2 = perm[i+1]
            t = p1
            new_idx[p1] = p2
            new_idx[p2] = t
        
        return batch[:, new_idx]
