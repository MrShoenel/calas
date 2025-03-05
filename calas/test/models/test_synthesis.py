import torch
from torch import nn, Tensor, device, cuda
from normflows.flows import Flow, AutoregressiveRationalQuadraticSpline, LULinearPermute
from calas.models.flow import CalasFlow, CalasFlowWithRepr
from calas.tools.two_moons import two_moons_rejection_sampling, two_moons_likelihood, pure_two_moons_likelihood
from calas.models.repr import ReconstructableRepresentation
from calas.models.synthesis import Linear
from typing import override, Sequence
from calas.test.models.test_flow import AE_UNet_Repr, make_flows
from pytest import mark
from calas.models.synthesis import Data2Data_Grad, Space, Likelihood



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
        b = flow.X_to_B(input=samp[0:batch_size], classes=samp_class[0:batch_size])[0]
        if torch.abs(b.mean() + 3) < .3:
            # Stop training, we're close enough now.
            break
    

    with Linear(flow=flow, space_in='X', space_out='E') as linear:
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

        worse_emb, worse_clz = linear.modify3(batch=holdout, classes=holdout_clz, target_lik=min_lik - 2*sd, condition='lower_than', return_all=True, u_min=0.0, u_max=0.3, max_steps=100)
        assert torch.all(flow.log_rel_lik_E(embeddings=worse_emb, classes=worse_clz) < min_lik)
        
        # worse_emb = linear.modify(sample=holdout, classes=holdout_clz, target_lik=min_lik - 2*sd, condition='lower_than', return_all=True, perc_change=0.05)


def test_Normal2Normal():
    from calas.models.synthesis import Normal2Normal

    torch.manual_seed(0)
    repr = AE_UNet_Repr(input_dim=2, hidden_sizes=(3,2,3))
    flow = CalasFlowWithRepr(num_classes=2, flows=make_flows(dim=repr.embed_dim), repr=repr).to(device=dev, dtype=dty)
    samp = two_moons_rejection_sampling(nsamples=100).to(device=dev, dtype=dty)
    samp_class = torch.full((100,), 0.).to(device=dev)

    n2n = Normal2Normal(flow=flow, seed=0)
    n2n.permute(embeddings=flow.X_to_E(samp), classes=samp_class, likelihood='decrease')


def test_GradientPerm():
    from calas.models.synthesis import GradientPerm_in_B

    torch.manual_seed(0)
    repr = AE_UNet_Repr(input_dim=2, hidden_sizes=(3,2,3))
    flow = CalasFlowWithRepr(num_classes=2, flows=make_flows(dim=repr.embed_dim), repr=repr).to(device=dev, dtype=dty)
    samp = two_moons_rejection_sampling(nsamples=100).to(device=dev, dtype=dty)
    samp_class = torch.full((100,), 0.).to(device=dev)

    gp = GradientPerm_in_B(flow=flow, seed=0, u_min=0.01, u_max=0.01)
    gp.permute(embeddings=flow.X_to_E(samp), classes=samp_class, likelihood='decrease')



def test_GradientQofB():
    from calas.models.synthesis import GradientPerm_in_QofB

    torch.manual_seed(0)
    repr = AE_UNet_Repr(input_dim=2, hidden_sizes=(3,2,3))
    flow = CalasFlowWithRepr(num_classes=2, flows=make_flows(dim=repr.embed_dim), repr=repr).to(device=dev, dtype=dty)
    samp = two_moons_rejection_sampling(nsamples=100).to(device=dev, dtype=dty)
    samp_class = torch.full((100,), 0.).to(device=dev)

    gqb = GradientPerm_in_QofB(flow=flow, seed=0, u_min=0.01, u_max=0.01, scale_grad=True)
    gqb.permute(embeddings=flow.X_to_E(samp), classes=samp_class, likelihood='increase')



def test_LinearPerm():
    from calas.models.synthesis import LinearPerm_in_B

    torch.manual_seed(0)
    repr = AE_UNet_Repr(input_dim=2, hidden_sizes=(3,2,3))
    flow = CalasFlowWithRepr(num_classes=2, flows=make_flows(dim=repr.embed_dim), repr=repr).to(device=dev, dtype=dty)
    samp = two_moons_rejection_sampling(nsamples=100).to(device=dev, dtype=dty)
    samp_class = torch.full((100,), 0.).to(device=dev)

    lp = LinearPerm_in_B(flow=flow, seed=0, u_min=0.001, u_max=0.001)
    lp.permute(embeddings=flow.X_to_E(samp), classes=samp_class, likelihood='decrease')



# def test_Normal2NormalGrad():
#     from calas.models.synthesis import Normal2NormalGrad

#     torch.manual_seed(0)
#     repr = AE_UNet_Repr(input_dim=2, hidden_sizes=(3,2,3))
#     flow = CalasFlowWithRepr(num_classes=2, flows=make_flows(dim=repr.embed_dim), repr=repr).to(device=dev, dtype=dty)
#     samp = two_moons_rejection_sampling(nsamples=100).to(device=dev, dtype=dty)
#     samp_class = torch.full((100,), 0.).to(device=dev)

#     n2ng = Normal2NormalGrad(flow=flow, seed=0, u_min=0.025, u_max=0.05)
#     n2ng.permute(embeddings=flow.X_to_E(samp), classes=samp_class, likelihood='decrease')


@mark.parametrize('space', [Space.Data, Space.Embedded, Space.Base])
@mark.parametrize('lik', [Likelihood.Increase, Likelihood.Decrease])
def test_PermuteData_in_space(space: Space, lik: Likelihood):
    from calas.models.synthesis import Data2Data_Grad

    torch.manual_seed(0)
    repr = AE_UNet_Repr(input_dim=2, hidden_sizes=(3,2,3))
    flow = CalasFlowWithRepr(num_classes=2, flows=make_flows(dim=repr.embed_dim), repr=repr).to(device=dev, dtype=dty)
    samp = two_moons_rejection_sampling(nsamples=100).to(device=dev, dtype=dty)
    samp_class = torch.full((100,), 0.).to(device=dev)

    lik_fn = flow.log_rel_lik_X if space == Space.Data else (flow.log_rel_lik_E if space == Space.Embedded else flow.log_rel_lik_B)
    if space == Space.Data:
        u_min, u_max = 0.01*samp.std(), 0.02*samp.std()
    elif space == Space.Embedded:
        u_min, u_max = 0.025, 0.05
        samp = flow.X_to_E(input=samp)
    else:
        u_min, u_max = 0.01, 0.02
        samp = flow.X_to_B(input=samp, classes=samp_class)[0]

    pdg = Data2Data_Grad(flow=flow, space=space, u_min=u_min, u_max=u_max)
    perm = pdg.permute(batch=samp, classes=samp_class, likelihood=lik)

    if space != Space.Data:
        # In data space, this often works, but not always
        if lik == Likelihood.Increase:
            assert torch.all(lik_fn(samp, samp_class) < lik_fn(perm, samp_class))
        else:
            assert torch.all(lik_fn(samp, samp_class) > lik_fn(perm, samp_class))


@mark.parametrize('lik', [Likelihood.Increase, Likelihood.Decrease])
def test_N2N_no_Grad(lik: Likelihood):
    from calas.models.synthesis import Normal2Normal_NoGrad

    torch.manual_seed(0)
    repr = AE_UNet_Repr(input_dim=2, hidden_sizes=(3,2,3))
    flow = CalasFlowWithRepr(num_classes=2, flows=make_flows(dim=repr.embed_dim), repr=repr).to(device=dev, dtype=dty)
    samp = two_moons_rejection_sampling(nsamples=100).to(device=dev, dtype=dty)
    samp_class = torch.full((100,), 0.).to(device=dev)
    samp_b = flow.X_to_B(input=samp, classes=samp_class)[0]

    n2n_ng = Normal2Normal_NoGrad(flow=flow, seed=0, u_min=0.01, u_max=0.01, stds_step_perc=0.025)
    perm_b = n2n_ng.permute(batch=samp_b, classes=samp_class, likelihood=lik)

    if lik == Likelihood.Increase:
        assert torch.all(flow.log_rel_lik_B(samp_b, samp_class) < flow.log_rel_lik_B(perm_b, samp_class))
    else:
        assert torch.all(flow.log_rel_lik_B(samp_b, samp_class) > flow.log_rel_lik_B(perm_b, samp_class))
    