import torch
from torch import Tensor
from typing import Self, Sequence
from ..data.permutation import Likelihood, Permute, SampleTooSmallException, Space, T



class Synthesis:
    def __init__(self, flow: T, space_in: Space, space_out: Space):
        self.perms: list[Permute[T]] = []
        self.flow = flow
        self.space_in = space_in
        self.space_out = space_out
    

    def add_permutation(self, perm: Permute[T]) -> Self:
        assert isinstance(perm, Permute)
        self.perms.append(perm)
        return self
    

    def add_permutations(self, *perms: Sequence[Permute[T]]) -> Self:
        for perm in perms:
            self.add_permutation(perm=perm)
        return self
    

    def log_rel_lik(self, sample: Tensor, classes: Tensor, space: Space) -> Tensor:
        if space == Space.Data:
            return self.flow.log_rel_lik_X(input=sample, classes=classes)
        elif space == Space.Embedded:
            return self.flow.log_rel_lik_E(embeddings=sample, classes=classes)
        elif space == Space.Base:
            return self.flow.log_rel_lik_B(base=sample, classes=classes)
        raise Exception(f'Space {space} not supported.')
    

    def space_2_space(self, sample: Tensor, classes: Tensor, old: Space, new: Space) -> Tensor:
        if old == new:
            return sample
        if old == Space.Data:
            if new == Space.Embedded:
                return self.flow.X_to_E(input=sample)
            elif new == Space.Base:
                return self.flow.X_to_B(input=sample, classes=classes)[0]
        elif old == Space.Embedded:
            if new == Space.Data:
                return self.flow.X_from_E(embeddings=sample)
            elif new == Space.Base:
                return self.flow.E_to_B(embeddings=sample, classes=classes)[0]
        elif old == Space.Base:
            if new == Space.Data:
                return self.flow.X_from_B(base=sample, classes=classes)[0]
            elif new == Space.Embedded:
                return self.flow.E_from_B(base=sample, classes=classes)[0]
        raise Exception(f'Going from Space {old} to Space {new} is not supported!')
    

    def synthesize(self, sample: Tensor, classes: Tensor, target_lik: float, target_lik_crit: float, likelihood: Likelihood, max_steps: int=20):
        assert len(self.perms) > 0
        clz_int = classes.squeeze().to(dtype=torch.int64)
        liks = self.log_rel_lik(sample=sample, classes=clz_int, space=self.space_in)

        results: list[Tensor] = []
        results_classes: list[Tensor] = []
        steps = 0
        old_space = self.space_in
        while steps < max_steps and sample.shape[0] > 0:
            steps += 1

            for perm in self.perms:
                sample_prime: Tensor = None
                # First make sure the to-permute sample is in the correct space!
                sample_prime = self.space_2_space(sample=sample, classes=clz_int, old=old_space, new=perm.space_in)
                old_space = perm.space_out
                with perm:
                    try:
                        sample_prime = perm.permute(batch=sample_prime, classes=clz_int, likelihood=likelihood)
                    except SampleTooSmallException:
                        continue
                sample_prime_liks = self.log_rel_lik(sample=sample_prime, classes=clz_int, space=perm.space_out)

                # Let's look at samples that went too far first and reset them.
                mask_critical = torch.where(sample_prime_liks < target_lik_crit, True, False) if likelihood == Likelihood.Decrease else torch.where(sample_prime_liks > target_lik_crit, True, False)

                if torch.any(mask_critical).item():
                    sample_prime[mask_critical] = sample[mask_critical]
                    sample_prime_liks[mask_critical] = liks[mask_critical]


                mask_replace = torch.where(sample_prime_liks < liks, True, False) if likelihood == Likelihood.Decrease else torch.where(sample_prime_liks > liks, True, False)

                if torch.any(mask_replace).item():
                    liks[mask_replace] = sample_prime_liks[mask_replace]
                    sample[mask_replace] = sample_prime[mask_replace]
                

                mask_accept = torch.where(sample_prime_liks < target_lik, True, False) if likelihood == Likelihood.Decrease else torch.where(sample_prime_liks > target_lik, True, False)
                
                if torch.any(mask_accept).item():
                    results.append(sample_prime[mask_accept])
                    results_classes.append(clz_int[mask_accept])
                    # Now remove these samples from the to-do list!
                    clz_int = clz_int[~mask_accept]
                    liks = liks[~mask_accept]
                    sample = sample[~mask_accept]

                if sample.shape[0] == 0:
                    break
        
        if len(results) == 0:
            return torch.empty(size=(0, sample.shape[1]), device=sample.device), torch.empty(size=(0,), device=classes.device)
        
        done = torch.vstack(tensors=results)
        done_classes = torch.cat(results_classes)
        done = self.space_2_space(sample=done, classes=done_classes, old=old_space, new=self.space_out)

        return done, done_classes
