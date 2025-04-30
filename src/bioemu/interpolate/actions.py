"""
From OMBasics/src/actions.py @ commit 6a8fbfa6
"""
import torch


class SimpleAction(torch.nn.Module):
    """
    Simple Onsager-Machlup action. For details, see Eqn 4 from the paper:
    Lee, J., Lee, IH., Joung, I. et al. Finding multiple reaction pathways via
    global optimization of action. Nat Commun 8, 15443 (2017). https://doi.org/10.1038/ncomms15443

    They provide a simplified Onsager-Machlup action, which avoids computing the Hessian of the energy and is thus more efficient.
    """

    def __init__(self, dt, gamma):
        """
        Args:
            dt: float, time step
            gamma: float, diffusion coefficient
        """
        super(SimpleAction, self).__init__()
        self.dt = dt
        self.gamma = gamma

    def forward(self, path: torch.Tensor, forces: torch.Tensor):
        """
        Args:
            path: torch.Tensor of images of shape [P, C, H, W], where P is the number of images on the path.
            forces: torch.Tensor of forces (derived from diffusion model score estimates) of shape [P, C, H, W], where N is the number of points on the path.
        Returns the OM action of the path (torch.Tensor of shape [1]).

        Note: we omit the term which involves the difference of the energy at the endpoints of the path,
        because it is constant and does not affect the optimization.
        """

        assert path.shape == forces.shape, "path and forces must have the same shape"
        assert len(path.shape) == 4, "path and forces must have shape [P, C, H, W]"

        result = 0.0
        first_term = torch.square(self.gamma / self.dt * (path[1:] - path[:-1]))

        second_term = (torch.square(forces[:-1]) + torch.square(forces[1:])) / 2.0

        third_term = (
            -self.gamma / self.dt * (path[1:] - path[:-1]) * (forces[:-1] - forces[1:])
        )
        result = torch.sum(first_term + second_term + third_term)

        return result * self.dt / (4 * self.gamma)


class HessianAction(torch.nn.Module):
    """
    Onsager-Machlup action involving the Hessian of the energy.
    See Eqn 7 from the paper:
    Artur B. Adib. Stochastic actions for diffusive dynamics: Reweight-
    ing, sampling, and minimization. The Journal of Physical Chemistry B,
    112(19):5910–5916, 05 2008

    Note: we omit the term which involves the difference of the energy at the endpoints of the path,
    because it is constant and does not affect the optimization.
    """

    def __init__(self, dt, xi, D):
        """
        Args:
            dt: float, time step
            xi: float, inverse temperature
            D: float, diffusion coefficient
        """
        super(HessianAction, self).__init__()
        self.dt = dt
        self.xi = xi
        self.D = D

    def forward(
        self, path: torch.Tensor, forces: torch.Tensor, force_grads: torch.Tensor
    ):
        """
        Args:
            path: torch.Tensor of images of shape [P, C, H, W], where P is the number of images on the path.
            forces: torch.Tensor of diffusion model score estimates of shape [P, C, H, W], where N is the number of points on the path.
            force_grads: torch.Tensor of gradients of the forces w.r.t. the path, of shape [P, C, H, W], where N is the number of points on the path.
            We only use the diagonal elements of the force grads. That is, how does the image at each point affect the force at that same point.
        Returns the OM action of the path (torch.Tensor of shape [1]).
        """
        # assert path.shape == forces.shape, "path and forces must have the same shape"
        # Path and forces have different shapes in the term-by-term case - path has shape [2, C, H, W] and forces has shape [1, C, H, W]
        assert len(path.shape) == 4, "path must have shape [P, C, H, W]"
        assert len(forces.shape) == 4, "forces must have shape [P, C, H, W]"
        assert len(force_grads.shape) == 4, "force grads must have shape [P, C, H, W]"

        assert (
            forces.shape == force_grads.shape
        ), "forces and force_grads must have the same shape"

        # add extra dimension to account for term-by-term case
        if forces[0].shape[0] == 1:
            forces = forces.repeat(2, 1, 1, 1)
            force_grads = force_grads.repeat(2, 1, 1, 1)

        first_term = torch.square(1 / self.dt * (path[1:] - path[:-1]))

        second_term = torch.square(forces[:-1] / self.xi)

        third_term = (
            -2 * self.D / self.xi * force_grads[:-1]
        )  # TODO: is there a minus sign in front of this term? (See Eqn 10 from https://pubs.aip.org/aip/jcp/article/132/13/134101/902772/Onsager-Machlup-action-based-path-sampling-and-its)
        # TODO Yes correct.
        result = torch.sum(first_term + second_term - third_term)

        return result * self.dt / 2


class TruncatedAction(torch.nn.Module):
    """
    Onsager-Machlup action, just ignoring the Hessian of energy.
    Artur B. Adib. Stochastic actions for diffusive dynamics: Reweight-
    ing, sampling, and minimization. The Journal of Physical Chemistry B,
    112(19):5910–5916, 05 2008

    Note: we omit the term which involves the difference of the energy at the endpoints of the path,
    because it is constant and does not affect the optimization.
    """

    def __init__(self, dt, xi):
        """
        Args:
            dt: float, time step
            xi: float, inverse temperature
            D: float, diffusion coefficient
        """
        super(TruncatedAction, self).__init__()
        self.dt = dt
        self.xi = xi

        # When ignoring Hessian, OM action is temperatureles
        # self.D = D

    def forward(self, path: torch.Tensor, forces: torch.Tensor):
        """
        Args:
            path: torch.Tensor of images of shape [P, C, H, W], where P is the number of images on the path.
            forces: torch.Tensor of diffusion model score estimates of shape [P, C, H, W], where N is the number of points on the path.
        Returns the OM action of the path (torch.Tensor of shape [1]).
        """
        # assert path.shape == forces.shape, "path and forces must have the same shape"
        # Path and forces have different shapes in the term-by-term case - path has shape [2, C, H, W] and forces has shape [1, C, H, W]
        assert len(path.shape) == 4, "path must have shape [P, C, H, W]"
        assert len(forces.shape) == 4, "forces must have shape [P, C, H, W]"

        # TODO ???
        # add extra dimension to account for term-by-term case
        # if forces[0].shape[0] == 1:
        #    forces = forces.repeat(2, 1, 1, 1)

        first_term = torch.square((path[1:] - path[:-1]) / self.dt)

        second_term = torch.square(forces[:-1] / self.xi)

        # Third term just ignored.

        result = torch.sum(first_term + second_term)

        return result * self.dt / 2.0