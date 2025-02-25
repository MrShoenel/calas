import torch
from normflows.distributions import ClassCondDiagGaussian
from normflows.flows import Flow, AutoregressiveRationalQuadraticSpline, LULinearPermute
from calas.models.flow import CalasFlow
from calas.tools.two_moons import two_moons_rejection_sampling, two_moons_likelihood, pure_two_moons_likelihood



def test_two_moons():
    samp = two_moons_rejection_sampling(nsamples=50, seed=0)
    lik = two_moons_likelihood(input=samp)
    lik_p = pure_two_moons_likelihood(input=samp)

    print(5)



def test_temp():
    q0 = ClassCondDiagGaussian(shape=10, num_classes=2)
    res = q0.forward(y=torch.tensor([0,1]))



def test_calas():
    K = 4

    latent_size = 2
    hidden_units = 128
    hidden_layers = 2
    ctx_channels = latent_size * 2 # µ + sd per feature

    flows: list[Flow] = []
    for _ in range(K):
        flows.append(AutoregressiveRationalQuadraticSpline(
            num_input_channels=latent_size, num_blocks=hidden_layers, num_hidden_channels=hidden_units, num_context_channels=ctx_channels))
        flows.append(LULinearPermute(num_channels=2))

    flow = CalasFlow(num_dims=2, num_classes=2, flows=flows)

    use_C = torch.tensor([0,0,0,0,0, 1,1,1,1,1])
    use_X = torch.rand((10,2))

    b, _ = flow.x_to_b(x=use_X, classes=use_C)
    x, _ = flow.b_to_x(b=b, classes=use_C)
    assert torch.allclose(input=use_X, other=x)


    samp, _, _ = flow.sample(5)

    loss = flow.loss(x=x, classes=use_C)

    print(5)
