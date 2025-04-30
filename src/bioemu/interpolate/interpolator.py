import torch
from bioemu.interpolate.om_lib import OMInterpolatorWrapper
from bioemu.interpolate import actions
from bioemu.datasets.fastfolders import FastFolderTrajectory

# class Interpolator(OMInterpolatorWrapper):
class Interpolator(torch.nn.Module):
    
    def __init__(self,
        # the user should set these via om_interpolate
        path_length, # N: number of points along path
        dt, # dt * path_length = T: total time of the path
        
        # om_interpolate should set this
        protein_trajectory: FastFolderTrajectory,

        # These user should set these in their config
        gamma=10, # in the actual script I am stepping through, this is a tensor with value 12.0108, shape (num_paths,)
        # set as samp_args.om_gamma * torch.tensor(masses).to(device)
        D=0.015, # set as samp_args.om_d / (trainset.std if args.scale_data else 1.0) ** 2
        # Specify parameters for the interpolator
        latent_time=0.01, # BioEmu ranges from 0.990 to 0.001 or smthg
        initial_guess_level=0.25, # BioEmu ranges from 0.990 to 0.001 or smthg
        # Specify parameters for the optimizer
        lr=2e-1,
        om_steps=500,
        path_batch_size=-1, # Turns off batch optimization--do the whole path

        # These I can set the right defaults and forget
        action_cls=actions.HessianAction,
        optimizer=torch.optim.Adam,
        device="cuda" if torch.cuda.is_available() else "cpu",

        # These are parts of the interface I probably don't need to support, just shove them into kwargs
        # or raise if changed from these defaults
        **kwargs,
        # temperature=1.0, # This is the temperature of what? Not the dynamics right? For sampling?
        # model=None,
        # initial_guess_fn=torch.slerp,
        # encode_and_decode=False, # bit mis-leading, this is whether to do optimization with datapoints at latent_time
        # # and then decode to t=eps instead of decoding them to eps, and optimizing them with the score from t_opt=latent_time
        # mlff=False,
        # cg_prior=False,
        # anneal=False, # What is this for
        # log=False, # see if this is logging during interpolation or log scale or smthg
        # add_noise=False, 
        # truncated_gradient=False, # What is this?
        # subsample_points_percent=None,
        # subsample_dimensions_percent=None, # During optimization?
        # sample_latent_time=False,
        # cosine_scheduler=False
    ):
        # the OMBasics codebase calls this gamma, but m * gamma is actually zeta
        super().__init__()
        self.gamma = gamma * torch.tensor(protein_trajectory.masses).to(device)
        self.D = D / protein_trajectory.std ** 2
        self.path_length = path_length

    def forward(self, x1, x2, z=None):
        return self.om_interpolate(x1, x2)

    def om_interpolate(self, x1, x2, **kwargs): # kwargs originally had z, the atom identities
        start_points_BRX = x1
        end_points_BRX = x2
        return {
            "final_path": torch.zeros_like(start_points_BRX) # TODO: implement this, for now just return zeros
        }
