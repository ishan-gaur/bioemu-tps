import torch
from bioemu.interpolate.actions import TruncatedAction
from dataclasses import dataclass, field
from typing import Optional, Callable, Type

import torch.optim
from bioemu.interpolate import actions

    
# Just have this in case I need to try to plug and play with other code they have later on
# that way my dictionaries will be formatted correctly
def sample_interpolations_from_model(
    interpolator,
    endpoint_1_samples,
    endpoint_2_samples,
    batch_size,
    verbose=False,
    z=None,
):
    """
    From OMBasics/two-for-one-diffusion/evaluators.py
    Sample interpolations from the model.
    """
    num_paths = endpoint_1_samples.shape[0] // torch.cuda.device_count()
    print(
        f"Generating {num_paths} interpolation paths per GPU. This may take some time."
    )
    all_path_list = []
    path_optimization_list = []
    all_actions_list = []
    all_path_terms_list = []
    all_force_terms_list = []
    endpoint_1_split = endpoint_1_samples.split(batch_size)
    endpoint_2_split = endpoint_2_samples.split(batch_size)
    for i, (x1, x2) in enumerate(zip(endpoint_1_split, endpoint_2_split)):
        output = interpolator(x1, x2, z)
        all_path_list.append(output["final_path"])

        if "all_paths" in output.keys():
            path_optimization_list.append(output["all_paths"])
        if "actions" in output.keys():
            all_actions_list.append(output["actions"])
        if "path_terms" in output.keys():
            all_path_terms_list.append(output["path_terms"])
        if "force_terms" in output.keys():
            all_force_terms_list.append(output["force_terms"])

        if verbose:
            print(f"Batch {i+1} from {len(endpoint_1_split)} generated")
    # all_mol_list = list(map(lambda n: model.sample(batch_size=n), batches))
    all_path = torch.cat(all_path_list, dim=0).cpu()

    output = {"sampled_mol": all_path}

    if len(path_optimization_list) > 0:
        all_paths = torch.cat(path_optimization_list, dim=1).cpu()
        all_actions = torch.cat(all_actions_list, dim=0).cpu()
        all_path_terms = torch.cat(all_path_terms_list, dim=0).cpu()
        all_force_terms = torch.cat(all_force_terms_list, dim=0).cpu()

        output.update(
            {
                "all_paths": all_paths,
                "actions": all_actions,
                "path_terms": all_path_terms,
                "force_terms": all_force_terms,
            }
        )

    print(f"{int(len(all_path) / interpolator.path_length)} paths generated")

    return output


# Just using this to make sure I stay as close to their interface as possible
class OMInterpolatorWrapper(torch.nn.Module):
    """
    From OMBasics/two-for-one-diffusion/utils.py at commit 6a8fbfa6
    The network becomes an OM interpolator, such that we can sample in parallel GPUs
    """

    def __init__(
        self,
        model,
        path_length,
        latent_time,
        encode_and_decode=True,
        mlff=False,
        cg_prior=False,
        action_cls=TruncatedAction,
        initial_guess_fn=torch.lerp,
        initial_guess_level=0,
        om_steps=100,
        optimizer=torch.optim.Adam,
        lr=2e-1,
        dt=0.1,
        gamma=10,
        D=0.01,
        path_batch_size=-1,
        anneal=False,
        sample_latent_time=False,
        cosine_scheduler=False,
        subsample_points_percent=None,
        subsample_dimensions_percent=None,
        add_noise=False,
        truncated_gradient=False,
        temperature=1.0,
        log=False,
    ):
        super(OMInterpolatorWrapper, self).__init__()
        self.model = model
        self.model.log = log
        self.path_length = path_length
        self.latent_time = latent_time
        self.encode_and_decode = encode_and_decode
        self.mlff = mlff
        self.cg_prior = cg_prior
        self.action_cls = action_cls
        self.initial_guess_fn = initial_guess_fn
        self.initial_guess_level = initial_guess_level
        self.om_steps = om_steps
        self.optimizer = optimizer
        self.lr = lr
        self.dt = dt
        self.gamma = gamma
        self.D = D
        self.path_batch_size = path_batch_size
        self.anneal = anneal
        self.sample_latent_time = sample_latent_time
        self.cosine_scheduler = cosine_scheduler
        self.subsample_points_percent = subsample_points_percent
        self.subsample_dimensions_percent = subsample_dimensions_percent
        self.add_noise = add_noise
        self.truncated_gradient = truncated_gradient
        self.temperature = temperature

    def forward(self, x1, x2, z=None):
        return self.model.om_interpolate(
            x1=x1,
            x2=x2,
            z=z,
            path_length=self.path_length,
            encode_and_decode=self.encode_and_decode,
            latent_time=self.latent_time,
            mlff=self.mlff,
            action_cls=self.action_cls,
            initial_guess_fn=self.initial_guess_fn,
            initial_guess_level=self.initial_guess_level,
            om_steps=self.om_steps,
            optimizer=self.optimizer,
            lr=self.lr,
            dt=self.dt,
            gamma=self.gamma,
            D=self.D,
            path_batch_size=self.path_batch_size,
            anneal=self.anneal,
            sample_latent_time=self.sample_latent_time,
            cosine_scheduler=self.cosine_scheduler,
            subsample_points_percent=self.subsample_points_percent,
            subsample_dimensions_percent=self.subsample_dimensions_percent,
            add_noise=self.add_noise,
            truncated_gradient=self.truncated_gradient,
            temperature=self.temperature,
        )