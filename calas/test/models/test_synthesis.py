import torch
from torch import nn, Tensor
from normflows.flows import Flow, AutoregressiveRationalQuadraticSpline, LULinearPermute
from calas.models.flow import CalasFlow, CalasFlowWithRepr
from calas.tools.two_moons import two_moons_rejection_sampling, two_moons_likelihood, pure_two_moons_likelihood
from calas.models.repr import RepresentationWithReconstruct
from calas.models.synthesis import Linear
from typing import override, Sequence
from calas.test.models.test_flow import AE_UNet_Repr, make_flows




def test_linear():
    repr = AE_UNet_Repr(input_dim=2, hidden_sizes=(32,16,32))
    assert repr.embed_dim == 80
    assert repr._decoder.in_features == sum((32,16,32)) and repr._decoder.out_features == 2

    flow = CalasFlowWithRepr(num_classes=2, flows=make_flows(dim=repr.embed_dim), repr=repr)

    samp = two_moons_rejection_sampling(nsamples=10)
    samp_class = torch.full((10,), 0.)

    with Linear(flow=flow) as linear:
        avg_lik = flow.log_rel_lik(input=samp, classes=samp_class).mean().item()

        conc = linear.modify(
            sample=samp, classes=samp_class, target_lik=avg_lik, condition='concentrate', concentrate_dim_log_tol=1.5)
        
        max_dev = repr.embed_dim * 1.5
        conc_lik = flow.log_rel_lik(input=conc, classes=samp_class)
        assert torch.all(max_dev > torch.abs(conc_lik - avg_lik))
        
        print(5)