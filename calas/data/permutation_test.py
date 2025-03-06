import torch
import pytest
from torch import device, cuda, Tensor
from calas.models.flow import CalasFlowWithRepr
from ..tools.two_moons import two_moons_rejection_sampling
from ..models.flow_test import make_flows
from ..models.repr import AE_UNet_Repr
from ..tools.func import normal_cdf, normal_ppf_safe
from .permutation import Space, Likelihood, LocsScales, GradientScaling, Data2Data_Grad, Normal2Normal_Grad, Normal2Normal_NoGrad, CurseOfDimDataPermute, PermuteDims
from pytest import mark
from typing import Callable, Literal



dev = device('cuda' if cuda.is_available() else 'cpu')
dty = torch.float32


from .synthesis_test import trained, untrained



@mark.parametrize('space', [Space.Data, Space.Embedded, Space.Base])
@mark.parametrize('lik', [Likelihood.Increase, Likelihood.Decrease], ids=['inc', 'dec'])
@mark.parametrize('fitted', [True, False], ids=['fit', 'not_fit'])
def test_D2D_Grad(space: Space, lik: Likelihood, fitted: bool):
    torch.manual_seed(0)
    flow, samp, samp_class = (trained if fitted else untrained).all

    lik_fn = flow.log_rel_lik_X if space == Space.Data else (flow.log_rel_lik_E if space == Space.Embedded else flow.log_rel_lik_B)
    if space == Space.Data:
        u_min, u_max = 0.01*samp.std(), 0.02*samp.std()
    elif space == Space.Embedded:
        u_min, u_max = 0.025, 0.05
        samp = flow.X_to_E(input=samp)
    else:
        u_min, u_max = 0.05, 0.05
        samp = flow.X_to_B(input=samp, classes=samp_class)[0]

    pdg = Data2Data_Grad(flow=flow, space=space, u_min=u_min, u_max=u_max)
    perm = pdg.permute(batch=samp, classes=samp_class, likelihood=lik)

    # This modification works best in E!
    thresh = 0.98 if space == Space.Embedded else (0.92 if space == Space.Base else 0.65)
    num_corr = (lik_fn(samp, samp_class) < lik_fn(perm, samp_class)).sum() if lik == Likelihood.Increase else (lik_fn(samp, samp_class) > lik_fn(perm, samp_class)).sum()
    
    assert (num_corr / samp.shape[0]) > thresh


@mark.parametrize('lik', [Likelihood.Increase, Likelihood.Decrease], ids=['inc', 'dec'])
@mark.parametrize('fitted', [True, False], ids=['fit', 'not_fit'])
@mark.parametrize('method', ['loc_scale', 'quantiles', 'hybrid'])
def test_N2N_Grad(lik: Likelihood, fitted: bool, method: Literal['loc_scale', 'quantiles', 'hybrid']):
    torch.manual_seed(0)
    flow, samp, samp_class = (trained if fitted else untrained).all
    samp_b = flow.X_to_B(input=samp, classes=samp_class)[0]

    n2ng = Normal2Normal_Grad(flow=flow, method=method, grad_scaling=GradientScaling.Softmax)
    perm_b = n2ng.permute(batch=samp_b, classes=samp_class, likelihood=lik)

    res = flow.log_rel_lik_B(samp_b, samp_class) < flow.log_rel_lik_B(perm_b, samp_class)
    if lik == Likelihood.Decrease:
        res = ~res
    assert (res.sum() / samp.shape[0]) > 0.999


@mark.parametrize('lik', [Likelihood.Increase, Likelihood.Decrease], ids=['inc', 'dec'])
@mark.parametrize('base_grad', [True, False], ids=['bg', 'no_bg'])
@mark.parametrize('fitted', [True, False], ids=['fit', 'not_fit'])
@mark.parametrize('method', ['loc_scale', 'quantiles'])
def test_N2N_NoGrad(lik: Likelihood, base_grad: bool, fitted: bool, method: Literal['loc_scale', 'quantiles']):
    torch.manual_seed(0)
    flow, samp, samp_class = (trained if fitted else untrained).all
    samp_b = flow.X_to_B(input=samp, classes=samp_class)[0]

    # NOTE: `u` determines the step size. Under a fitted model and without
    # a guiding (base) gradient, this step size should be much smaller, or
    # otherwise, this move naiive method will overshoot and fail to confidently
    # push the likelihood towards the desired direction.
    use_u = 0.001 if fitted and not base_grad else 0.01
    n2n_ng = Normal2Normal_NoGrad(flow=flow, seed=0, u_min=use_u, u_max=use_u, stds_step_perc=0.025, use_loc_scale_base_grad=base_grad, method=method, grad_scaling=GradientScaling.Normalize)
    perm_b = n2n_ng.permute(batch=samp_b, classes=samp_class, likelihood=lik)

    num_corr = (flow.log_rel_lik_B(samp_b, samp_class) < flow.log_rel_lik_B(perm_b, samp_class)).sum() if lik == Likelihood.Increase else (flow.log_rel_lik_B(samp_b, samp_class) > flow.log_rel_lik_B(perm_b, samp_class)).sum()

    assert num_corr == samp.shape[0]
        

@mark.parametrize('use_grad_dir', [True, False], ids=['grad', 'no_grad'])
@mark.parametrize('fitted', [True, False], ids=['fit', 'not_fit'])
@mark.parametrize('space', [Space.Data, Space.Embedded, Space.Base, Space.Quantiles])
def test_CurseOfDim(use_grad_dir: bool, space: Space, fitted: bool):
    torch.manual_seed(0)
    flow, samp, samp_class = (trained if fitted else untrained).all

    if space == Space.Quantiles and use_grad_dir:
        with pytest.raises(Exception):
            CurseOfDimDataPermute(flow=flow, space=space, use_grad_dir=use_grad_dir)
        return
    
    
    lik_fn: Callable[[Tensor, Tensor], Tensor] = flow.log_rel_lik_X
    if space == Space.Embedded:
        lik_fn = flow.log_rel_lik_E
        samp = flow.X_to_E(input=samp)
    elif space == Space.Base:
        lik_fn = flow.log_rel_lik_B
        samp = flow.X_to_B(input=samp, classes=samp_class)[0]
    elif space == Space.Quantiles:
        temp = Normal2Normal_NoGrad(flow=flow, method='quantiles', locs_scales_flow_frac=0.0, locs_scales_mode=LocsScales.Averaged)
        samp = flow.X_to_B(input=samp, classes=samp_class)[0]
        means, stds = temp.averaged_locs_and_scales(batch=samp, classes=samp_class)
        samp = torch.vmap(normal_cdf)(samp, means, stds)
        lik_fn = lambda q, clz: flow.log_rel_lik_B(base=torch.vmap(normal_ppf_safe)(q, means, stds), classes=clz)

    num_dims = (1,2) if space == Space.Data else (4,8)
    codp = CurseOfDimDataPermute(flow=flow, space=space, use_grad_dir=use_grad_dir, num_dims=num_dims, grad_scaling=GradientScaling.NoScale)
    perm = codp.permute(batch=samp, classes=samp_class)
    assert samp.shape == perm.shape

    before, after = lik_fn(samp, samp_class), lik_fn(perm, samp_class)
    num_worse = torch.where(after < before, 1., 0).sum().item()

    if space == Space.Quantiles:
        # Usually, in quantile space, the samples become better. This makes sense,
        # because the manipulated quantiles tend to go towards the mean on average,
        # where the resulting likelihood is higher.
        assert (num_worse / samp.shape[0]) < 0.66
    elif space == Space.Data:
        # There are only 2 dimensions here..
        assert (num_worse / samp.shape[0]) >= 0.39
    elif space == Space.Embedded:
        assert (num_worse / samp.shape[0]) > 0.60
    elif space == Space.Base:
        assert (num_worse / samp.shape[0]) > 0.43


def test_PermuteDims():
    torch.manual_seed(0)
    repr = AE_UNet_Repr(input_dim=2, hidden_sizes=(8,4,8))
    flow = CalasFlowWithRepr(num_classes=2, flows=make_flows(dim=repr.embed_dim), repr=repr).to(device=dev, dtype=dty)
    samp = two_moons_rejection_sampling(nsamples=100).to(device=dev, dtype=dty)
    samp_class = torch.full((100,), 0.).to(device=dev)
    samp_b = flow.X_to_E(samp)

    with pytest.warns(Warning):
        pd = PermuteDims(flow=flow, space=Space.Base, num_dims=(10,flow.num_dims), change_prob=0.999)
        perm_b = pd.permute(batch=samp_b)

        # For a normalizing flow, permuting dimensions in E or B has *no* effect (!!)
        before, after = flow.log_rel_lik_B(samp_b, samp_class), flow.log_rel_lik_B(perm_b, samp_class)
        assert torch.allclose(before, after, atol=1e-4)


    pdx = PermuteDims(flow=flow, space=Space.Data, num_dims=(2,2), change_prob=1.0)
    perm = pdx.permute(batch=samp)

    before, after = flow.log_rel_lik_X(samp, samp_class), flow.log_rel_lik_X(perm, samp_class)
    # There should be quite a big difference!
    assert torch.abs(before - after).min() > 1.
