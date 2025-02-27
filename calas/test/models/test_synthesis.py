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


def test_linear_e2e():
    """
    We can only really test this feature if the flow has been trained somewhat.
    Here, we will train the flow until most of the data is where it should be
    in the B space. Then, we'll test the linear synthesis.
    """

    torch.manual_seed(0)
    repr = AE_UNet_Repr(input_dim=2, hidden_sizes=(4))
    # assert repr.embed_dim == 32
    assert repr._decoder.out_features == 2

    flow = CalasFlowWithRepr(num_classes=2, flows=make_flows(dim=repr.embed_dim), repr=repr).to(device=dev, dtype=dty)

    batch_size, n_samp, n_holdout = 100, 1_000, 64
    samp = two_moons_rejection_sampling(nsamples=n_samp + n_holdout).to(device=dev, dtype=dty)
    samp_class = torch.full((n_samp + n_holdout,), 0.).to(device=dev)

    # Let's optimize the parameters of the flow AND its representation!
    assert len(list(flow.parameters(recurse=False))) < len(list(flow.parameters(recurse=True)))
    optim = torch.optim.Adam(params=flow.parameters(recurse=True), lr=1e-3)
    while True:
        optim.zero_grad()
        perm = torch.randperm(n_samp - n_holdout, device=dev) # Only use first 80 % for train
        loss = flow.loss(input=samp[perm], classes=samp_class[perm])
        loss.backward()
        optim.step()

        # The goal is for the flow to learn to push the class-0 (ID) data towards
        # its conditional mean in the B-space, which is at -3! We want to stop
        # this training if the training data is (at least mostly) close to that.
        b = flow.x_to_b(input=samp[0:batch_size], classes=samp_class[0:batch_size])[0]
        if torch.abs(b.mean() + 3) < .3:
            # Stop training, we're close enough now.
            break
    

    with Linear(flow=flow) as linear:
        holdout, holdout_clz = samp[n_samp:], samp_class[n_samp:]
        temp = flow.log_rel_lik(input=holdout, classes=holdout_clz)
        avg_lik = temp.mean()
        sd = temp.std()

        # conc = linear.modify(sample=holdout, classes=holdout_clz, target_lik=avg_lik, condition='concentrate', concentrate_dim_log_tol=sd / 5, return_all=True) # Concentrated likelihoods must be around the average with a maximum of a 5th of a std deviation away from it.
        # assert torch.all(sd/5 > torch.abs(flow.log_rel_lik(input=conc, classes=holdout_clz) - avg_lik))


        # NEXT TEST: Make samples worse! That should work without any problems,
        # because the flow is somewhat trained and, therefore, we should be able
        # to essentially push samples almost arbitrarily far away!
        min_lik = temp.min()
        
        worse = linear.modify(sample=holdout, classes=holdout_clz, target_lik=min_lik - 2*sd, condition='lower_than', return_all=True, perc_change=0.05)
        assert torch.all(flow.log_rel_lik(input=worse, classes=holdout_clz) < min_lik)
