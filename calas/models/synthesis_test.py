import torch
from torch import device, cuda, Tensor
from .synthesis import Synthesis
from ..data.permutation import Space, Likelihood, Data2Data_Grad, Normal2Normal_Grad, Normal2Normal_NoGrad, CurseOfDimDataPermute
from .flow import CalasFlowWithRepr
from .flow_test import make_flows
from .repr import AE_UNet_Repr
from ..tools.two_moons import two_moons_rejection_sampling
from ..tools.lazy import Lazy
from pytest import mark, fixture


dev = device('cuda' if cuda.is_available() else 'cpu')
dty = torch.float32


class Trained:
    def __init__(self, max_steps: int, seed: int=1):
        self.trained: Lazy[tuple[CalasFlowWithRepr, Tensor, Tensor]] = Lazy(factory=lambda: Trained.train_flow(seed=seed, max_steps=max_steps))
    
    @property
    def all(self) -> tuple[CalasFlowWithRepr, Tensor, Tensor]:
        return self.trained.value
    
    @property
    def flow(self) -> CalasFlowWithRepr:
        return self.all[0]
    
    @property
    def testdata(self) -> Tensor:
        return self.all[1]
    
    @property
    def testdata_clz(self) -> Tensor:
        return self.all[2]
    
    @staticmethod
    def train_flow(seed: int=1, max_steps: int=10) -> tuple[CalasFlowWithRepr, Tensor, Tensor]:
        """
        Returns a (somewhat) trained flow and repr, together with some holdout-data
        and -classes.
        """
        torch.manual_seed(seed=seed)
        repr = AE_UNet_Repr(input_dim=2, hidden_sizes=(3,2,3))
        assert repr._decoder.out_features == 2

        flow = CalasFlowWithRepr(num_classes=2, flows=make_flows(dim=repr.embed_dim), repr=repr).to(device=dev, dtype=dty)

        n_samp, n_holdout = 1_000, 64
        samp = two_moons_rejection_sampling(nsamples=n_samp + n_holdout).to(device=dev, dtype=dty)
        samp_class = torch.full((n_samp + n_holdout,), 0.).to(device=dev)

        # Let's optimize the parameters of the flow AND its representation!
        assert len(list(flow.parameters(recurse=False))) < len(list(flow.parameters(recurse=True)))
        optim = torch.optim.Adam(params=flow.parameters(recurse=True), lr=5e-4)
        steps = 0
        loss_before = float('inf')
        while steps < max_steps:
            steps += 1
            optim.zero_grad()
            perm = torch.randperm(n_samp - n_holdout, device=dev) # Only use first 80 % for train
            loss = flow.loss_wrt_X(input=samp[perm], classes=samp_class[perm])
            loss.backward()
            optim.step()

            if loss.item() > loss_before:
                break
            loss_before = loss.item()
        
        return flow, samp[n_samp:], samp_class[n_samp:]


untrained = Trained(max_steps=0)
trained = Trained(max_steps=10)



@mark.parametrize('lik', [Likelihood.Increase, Likelihood.Decrease], ids=['inc', 'dec'])
@mark.parametrize('fitted', [True, False], ids=['fit', 'not_fit'])
def test_Synthesis_e2e(lik: Likelihood, fitted: bool, space: Space=Space.Embedded):
    torch.manual_seed(0)
    flow, holdout, samp_clz = (trained if fitted else untrained).all
    holdout, samp_clz = holdout[0:20], samp_clz[0:20]
    samp_E = flow.X_to_E(input=holdout)

    synth = Synthesis(flow=flow, space_in=space, space_out=space)

    synth.add_permutation(Data2Data_Grad(flow=flow, space=space, u_min=0.1, u_max=0.15))
    # synth.add_permutation(Normal2Normal_Grad(flow=flow, method='hybrid', u_min=1.0, u_max=1.0))
    synth.add_permutation(CurseOfDimDataPermute(flow=flow, space=space, num_dims=(5,15), use_grad_dir=False))
    # synth.add_permutation(Normal2Normal_NoGrad(flow=flow, method='quantiles', u_min=0.05, u_max=0.05))

    liks_before = synth.log_rel_lik(sample=samp_E, classes=samp_clz, space=space)
    target_lik = liks_before.min() + (1. if lik == Likelihood.Increase else -1.) * 5 * liks_before.std()
    target_lik_crit = target_lik + (1. if lik == Likelihood.Increase else -1.) * 15 * liks_before.std()

    samp_E_prime, samp_prime_clz = synth.synthesize(sample=samp_E, classes=samp_clz, target_lik=target_lik, target_lik_crit=target_lik_crit, likelihood=lik, max_steps=10)
    liks_after = synth.log_rel_lik(sample=samp_E_prime, classes=samp_prime_clz, space=space)

    # Perhaps we get back fewer samples!
    assert samp_E.shape[0] >= int(0.95 * samp_E.shape[0])

    if lik == Likelihood.Decrease:
        assert torch.all(liks_after <= target_lik)
        assert torch.all(liks_after >= target_lik_crit)
    else:
        assert torch.all(liks_after >= target_lik)
        assert torch.all(liks_after <= target_lik_crit)
