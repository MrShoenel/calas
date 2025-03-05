import torch
from normflows.flows import Flow, AutoregressiveRationalQuadraticSpline, LULinearPermute
from calas.models.flow import CalasFlow, CalasFlowWithRepr
from calas.tools.two_moons import two_moons_rejection_sampling, two_moons_likelihood, pure_two_moons_likelihood
from calas.models.repr import AE_UNet_Repr



def make_flows(K: int=4, dim: int=2, units: int=128, layers: int=2) -> list[Flow]:
    # ctx_channels = dim * 2, because µ + sd per feature
    flows: list[Flow] = []
    for _ in range(K):
        flows.append(AutoregressiveRationalQuadraticSpline(
            num_input_channels=dim, num_blocks=layers, num_hidden_channels=units, num_context_channels=2*dim))
        flows.append(LULinearPermute(num_channels=dim))
    return flows



def test_two_moons():
    samp = two_moons_rejection_sampling(nsamples=50, seed=0)
    lik = two_moons_likelihood(input=samp)
    lik_p = pure_two_moons_likelihood(input=samp)



def test_calas():
    flow = CalasFlow(num_dims=2, num_classes=2, flows=make_flows())

    use_C = torch.tensor([0,0,0,0,0, 1,1,1,1,1])
    use_X = torch.rand((10,2))

    b, _ = flow.E_to_B(embeddings=use_X, classes=use_C)
    x, _ = flow.E_from_B(base=b, classes=use_C)
    assert torch.allclose(input=use_X, other=x)


    samp, _, _ = flow.sample_E(5)
    assert samp.shape == (5,2)

    loss = flow.loss(input=x, classes=use_C)


def test_calas_repr():
    repr = AE_UNet_Repr(input_dim=2, hidden_sizes=(32,16,32))
    assert repr.embed_dim == 80
    assert repr._decoder.in_features == sum((32,16,32)) and repr._decoder.out_features == 2

    flow = CalasFlowWithRepr(num_classes=2, flows=make_flows(dim=repr.embed_dim), repr=repr)
    
    samp = two_moons_rejection_sampling(nsamples=100)
    samp_classes = torch.where(samp[:, 0] < 0, 0, 1)

    embeddings = flow.repr.embed(x=samp)
    assert embeddings.shape == (100, repr.embed_dim)
    
    samp2 = flow.sample_E(n_samp=50)[0]
    assert samp2.shape == (50, sum((32,16,32)))
    samp3 = flow.sample(n_samp=10)[0]
    assert samp3.shape == (10, 2)


def test_spaces():
    torch.manual_seed(0)
    repr = AE_UNet_Repr(input_dim=2, hidden_sizes=(32,16,32))
    flow = CalasFlowWithRepr(num_classes=2, flows=make_flows(dim=repr.embed_dim), repr=repr)
    
    samp = two_moons_rejection_sampling(nsamples=10)
    samp_class = torch.full((10,), 0.)
    
    e  = flow.X_to_E(input=samp)
    b  = flow.E_to_B(embeddings=e, classes=samp_class)[0]
    e1 = flow.E_from_B(base=b, classes=samp_class)[0]
    x1 = flow.X_from_E(embeddings=e1)

    assert torch.allclose(input=e, other=e1, atol=1e-7)
    # WE CANNOT assert the following because the representation was not trained to reconstruct properly!
    assert not torch.allclose(input=samp, other=x1) 
    
    # # Most likely, we cannot create anomalies with lower likelihood at this
    # # point, simply because the flow is not yet trained and will therefore
    # # assign very low likelihoods to the given ID samples.
    # lower_lik = flow.make_linear_global_anomaly(input=samp, classes=samp_class, likelihood='decrease')

    # # However, that here should work!
    # higher_lik = flow.make_linear_global_anomaly(input=samp, classes=samp_class, likelihood='increase')
    # samp_lik = flow.log_rel_lik(input=samp, classes=samp_class)
    # assert torch.all(samp_lik < flow.log_rel_lik(input=higher_lik, classes=samp_class))
