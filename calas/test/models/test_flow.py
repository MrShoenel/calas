import torch
from torch import nn, Tensor
from normflows.flows import Flow, AutoregressiveRationalQuadraticSpline, LULinearPermute
from calas.models.flow import CalasFlow, CalasFlowWithRepr
from calas.tools.two_moons import two_moons_rejection_sampling, two_moons_likelihood, pure_two_moons_likelihood
from calas.models.repr import RepresentationWithReconstruct
from typing import override, Sequence



def make_flows(K: int=4, dim: int=2, units: int=128, layers: int=2) -> list[Flow]:
    # ctx_channels = dim * 2, because µ + sd per feature
    flows: list[Flow] = []
    for _ in range(K):
        flows.append(AutoregressiveRationalQuadraticSpline(
            num_input_channels=dim, num_blocks=layers, num_hidden_channels=units, num_context_channels=2*dim))
        flows.append(LULinearPermute(num_channels=dim))
    return flows



class AE_UNet_Repr(RepresentationWithReconstruct):
    def __init__(self, input_dim, hidden_sizes: Sequence[int], *args, **kwargs):
        super().__init__(input_dim, *args, **kwargs)

        self.hidden_sizes = hidden_sizes

        self.hidden_modules: list[nn.Sequential] = []
        for idx in range(len(hidden_sizes)):
            num_in = input_dim if idx == 0 else hidden_sizes[idx-1]
            num_out = hidden_sizes[idx]

            mods: list[nn.Module] = []
            if idx > 0:
                mods.append(nn.Dropout1d(p=0.2))
            mods.append(nn.Linear(in_features=num_in, out_features=num_out, bias=True))
            mods.append(nn.SiLU())

            self.hidden_modules.append(nn.Sequential(*mods))

        self._decoder = nn.Linear(in_features=self.embed_dim, out_features=self.input_dim, bias=True)
    

    @property
    @override
    def embed_dim(self) -> int:
        return sum(self.hidden_sizes)
    
    @property
    @override
    def decoder(self) -> nn.Module:
        return self._decoder

    @override
    def forward(self, x: Tensor) -> Tensor:
        reprs: list[Tensor] = []
        prev: Tensor = x
        for seq in self.hidden_modules:
            reprs.append(seq(prev))
            prev = reprs[-1]
        
        return torch.hstack(reprs)
    


def test_two_moons():
    samp = two_moons_rejection_sampling(nsamples=50, seed=0)
    lik = two_moons_likelihood(input=samp)
    lik_p = pure_two_moons_likelihood(input=samp)



def test_calas():
    flow = CalasFlow(num_dims=2, num_classes=2, flows=make_flows())

    use_C = torch.tensor([0,0,0,0,0, 1,1,1,1,1])
    use_X = torch.rand((10,2))

    b, _ = flow.x_to_b(x=use_X, classes=use_C)
    x, _ = flow.b_to_x(b=b, classes=use_C)
    assert torch.allclose(input=use_X, other=x)


    samp, _, _ = flow.sample(5)
    assert samp.shape == (5,2)

    loss = flow.loss(x=x, classes=use_C)


def test_calas_repr():
    repr = AE_UNet_Repr(input_dim=2, hidden_sizes=(56,16,56))
    assert repr.embed_dim == 128

    flow = CalasFlowWithRepr(num_classes=2, flows=make_flows(dim=repr.embed_dim), repr=repr)
    
    samp = two_moons_rejection_sampling(nsamples=100)
    samp_classes = torch.where(samp[:, 0] < 0, 0, 1)

    embeddings = flow.repr.forward(x=samp)
    assert embeddings.shape == (100, repr.embed_dim)
    
    samp2 = flow.sample(n_samp=50)[0]
    assert samp2.shape == (50, 2)


    samp_cod = flow.make_CoD_batch_random_embedding(
        nominal=samp, classes=samp_classes, distr=None, num_dims=(8,8), mode=None, use_grad_dir=True, normalize=True)
    assert samp_cod.shape == (samp.shape[0], repr.embed_dim)

    classes_nominal = torch.zeros(samp.shape[0], device=samp.device)
    classes_anomaly = torch.ones(samp_cod.shape[0], device=samp_cod.device)
    total_loss =\
        flow.loss(x=samp, classes=classes_nominal) +\
        flow.loss_embedded(embedded=samp_cod, classes=classes_anomaly) +\
        repr.loss(x=samp)
    
    # The first loss is about the nominal samples under the nominal class. It also
    # affects the representation because loss() calls self.repr.forward(...).
    # The second loss does not affect the representation, because we are forwarding
    # a sample that was already embedded and has previously been *detached* from
    # the computational graph of the representation. In other words, in this case,
    # the anomalous samples will not affect how the representation trains and it
    # will *not* learn to reconstruct those inputs.
    # The third loss is an extra output in which the representation learns to re-
    # construct its inputs. All three loses here are KL divergences so we can add
    # them together.
    
    print(total_loss)

    # In the following, we change the first loss in order to *not* affect the re-
    # presentation. While we are using it to create an embedding of our nominal
    # batch, we are deliberately detaching it. Now the only loss that really has
    # an impact on the representation is its own loss (here: the 3rd).
    total_loss =\
        flow.loss_embedded(embedded=repr.forward(x=samp).detach(), classes=classes_nominal) +\
        flow.loss_embedded(embedded=samp_cod, classes=classes_anomaly) +\
        repr.loss(x=samp)
    
    
    # In this example, the reconstruction will actually learn to reconstruct the
    # anomalous samples, because we are using it directly to reconstruct a sample
    # that we have deliberately made worse before. However, note the *detach()*!
    # Like in the other 3-way losses, the first two losses give us the contrast
    # in shape of conditions that are then applied to the base distribution.
    total_loss =\
        flow.loss(x=samp, classes=classes_nominal) +\
        flow.loss(x=repr.reconstruct(embeddings=samp_cod).detach(), classes=classes_anomaly) +\
        repr.loss(x=samp)

    
    
