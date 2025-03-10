import torch
from torch import Tensor
from typing import Self, Sequence, Optional
from ..data.permutation import Likelihood, Permute, SampleTooSmallException, Space, T
from ..tools.mixin import NoGradNoTrainMixin



class Synthesis(NoGradNoTrainMixin[T]):
    def __init__(self, flow: T, space_in: Space, space_out: Space):
        NoGradNoTrainMixin.__init__(self=self, module=flow)
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
    

    def rsynthesize(self, sample: Tensor, classes: Tensor, likelihood: Likelihood, target_lik: float, target_lik_crit: Optional[float]=None, max_steps: int=20, accept_all: bool=False) -> tuple[Tensor, Tensor]:
        """
        Bla.
        """

        assert len(self.perms) > 0, 'No permutations have been configured.'
        assert not sample.requires_grad and not classes.requires_grad, 'The variant here uses the reparameterization trick and returns a tensor that can be added to the original sample.'

        with self:
            clz_int = classes.squeeze().to(dtype=torch.int64)
            # Transfer the sample already into the space of the first permutation.
            liks = self.log_rel_lik(sample=sample, classes=clz_int, space=self.perms[0].space_in)
            num_obs = sample.shape[0]
            
            # Let's reserve a tensor for the reparameterization trick in the *output*-space:
            sample_prime_idx = torch.tensor(list(range(num_obs)), device=sample.device)

            result_list: list[Tensor] = []
            result_list_idx: list[Tensor] = []

            steps = 0
            old_space = self.space_in
            while steps < max_steps and sample.shape[0] > 0:
                steps += 1

                for perm in self.perms:
                    with perm:
                        sample_prime = self.space_2_space(sample=sample, classes=clz_int, old=old_space, new=perm.space_in)
                        try:
                            sample_prime = perm.permute(batch=sample_prime, classes=clz_int, likelihood=likelihood)
                        except SampleTooSmallException:
                            continue

                        sample_prime_liks = self.log_rel_lik(sample=sample_prime, classes=clz_int, space=perm.space_out)
                        if not target_lik_crit is None:
                            # Let's look at samples that went too far first and reset them.
                            mask_critical = torch.where(sample_prime_liks < target_lik_crit, True, False) if likelihood == Likelihood.Decrease else torch.where(sample_prime_liks > target_lik_crit, True, False)

                            if torch.any(mask_critical).item():
                                # Reset those samples to what they were before.
                                sample_prime[mask_critical] = sample[mask_critical]
                                sample_prime_liks[mask_critical] = liks[mask_critical]
                        

                        mask_replace = torch.where(sample_prime_liks < liks, True, False) if likelihood == Likelihood.Decrease else torch.where(sample_prime_liks > liks, True, False)
                        if torch.any(mask_replace).item():
                            # Already replace samples that went into the right direction.
                            liks[mask_replace] = sample_prime_liks[mask_replace]
                            sample[mask_replace] = sample_prime[mask_replace]
                        

                        mask_accept = torch.where(sample_prime_liks < target_lik, True, False) if likelihood == Likelihood.Decrease else torch.where(sample_prime_liks > target_lik, True, False)
                        if torch.any(mask_accept).item():
                            result_list.append(self.space_2_space(sample=sample_prime[mask_accept], classes=clz_int[mask_accept], old=perm.space_out, new=self.space_out))
                            result_list_idx.append(sample_prime_idx[mask_accept])

                            # Now remove these samples from the to-do list!
                            clz_int = clz_int[~mask_accept]
                            sample_prime_idx = sample_prime_idx[~mask_accept]
                            sample = sample[~mask_accept]
                            liks = liks[~mask_accept]

                        if sample.shape[0] == 0:
                            break
            

            output_dims = self.flow.num_dims_X if self.space_out == Space.Data else self.flow.num_dims_E # Note E==B, so no more ifs required.
            result = torch.zeros(size=(num_obs, output_dims), device=sample.device, dtype=sample.dtype)
            result_idx_mask = torch.full(size=(num_obs,), fill_value=False)


            if len(result_list_idx) > 0:
                result_idx = torch.cat(tensors=result_list_idx)
                result[result_idx] = torch.vstack(tensors=result_list)
                result_idx_mask[result_idx] = True
            
            if accept_all and sample.shape[0] > 0:
                result[sample_prime_idx] = sample
                

            return result, result_idx_mask



    def synthesize(self, sample: Tensor, classes: Tensor, likelihood: Likelihood, target_lik: float, target_lik_crit: Optional[float]=None, max_steps: int=20):
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
                with torch.no_grad(): # TODO: THIS SHOULD BE OPTIONAL!
                    # First make sure the to-permute sample is in the correct space!
                    sample_prime = self.space_2_space(sample=sample, classes=clz_int, old=old_space, new=perm.space_in)
                
                old_space = perm.space_out
                with perm:
                    try:
                        sample_prime = perm.permute(batch=sample_prime, classes=clz_int, likelihood=likelihood) # TODO: Check if this is what we want in case we want to differentiate through this!
                    except SampleTooSmallException:
                        continue
                
                sample_prime_liks: Tensor = None
                with torch.no_grad():
                    sample_prime_liks = self.log_rel_lik(sample=sample_prime, classes=clz_int, space=perm.space_out)

                if not target_lik_crit is None:
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
