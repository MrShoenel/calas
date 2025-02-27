import torch
from torch import nn, Tensor, device, cuda
from normflows.flows import Flow, AutoregressiveRationalQuadraticSpline, LULinearPermute
from calas.models.flow import CalasFlow, CalasFlowWithRepr
from calas.tools.two_moons import two_moons_rejection_sampling, two_moons_likelihood, pure_two_moons_likelihood
from calas.models.repr import ReconstructableRepresentation
from calas.models.synthesis import Linear
from typing import override, Sequence
from calas.test.models.test_flow import AE_UNet_Repr, make_flows



dev = device('cuda' if cuda.is_available() else 'cpu')
dty = torch.float32



def test_linear():
    torch.manual_seed(0)
    repr = AE_UNet_Repr(input_dim=2, hidden_sizes=(3,2,3))
    repr.eval() # Put in eval mode, so we get deterministic behavior (has dropout!)
    assert all(not p.training for p in repr.__dict__.values() if isinstance(p, nn.Module))
    assert repr.embed_dim == 8
    assert repr._decoder.out_features == 2

    flow = CalasFlowWithRepr(num_classes=2, flows=make_flows(dim=repr.embed_dim), repr=repr)
    flow.to(device=dev, dtype=dty)

    samp = two_moons_rejection_sampling(nsamples=10).to(device=dev, dtype=dty)
    samp_class = torch.full((10,), 0.).to(device=dev, dtype=dty)

    with Linear(flow=flow) as linear:
        avg_lik = flow.log_rel_lik(input=samp, classes=samp_class).mean()

        conc = linear.modify(
            sample=samp, classes=samp_class, target_lik=avg_lik, condition='concentrate', concentrate_dim_log_tol=1.5,
            return_all=True)
        
        max_dev = repr.embed_dim * 1.5
        conc_lik = flow.log_rel_lik(input=conc, classes=samp_class)
        assert torch.all(max_dev > torch.abs(conc_lik - avg_lik))
        
        print(5)