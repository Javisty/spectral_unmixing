"""Define the regularization terms."""
import torch
import torch.nn as nn
import numpy as np


class AbundanceRegularization(nn.Module):
    """Regularization on abundances."""

    def __init__(self, t_barrier, spoq_params, sparsity_penalty):
        """
        t_barrier: parameter to define the log-barrier function.
        spoq_params: (p, q, alpha, beta, eta) the tuple of parameters defining
        the pseudo-norm used for sparsity
        sparsity_penalty: penalty on the sparsity term
        """
        super().__init__()
        self.t = t_barrier
        self.spoqs = spoq_params
        self.delta = sparsity_penalty

    def forward(self, A):
        """Return the regularization term for abundances A."""
        pass


class DispersionRegularization(nn.Module):
    """Regularization on Dispersion model parameters."""
    def __init__(self, penalty):
        """penalty: the penalty on the regularization term."""
        super().__init__()
        self.penalty = penalty

    def forward(self, theta):
        """Return the regularization term for parameters theta."""
        pass


class SPOQ(nn.Module):
    """
    Pseudo-norm SPOQ.

    See paper DOI:10.1109/TSP.2020.3025731 (Cherni 2020)
    """

    def __init__(self, p, q, alpha, beta, eta):
        """Parameters for the pseudo-norm."""
        super().__init()
        self.p = p
        self.q = q
        self.alpha = alpha
        self.beta = beta
        self.eta = eta

    def forward(self, A):
        """Return the pseudo-norm for A."""
        pass


class LogBarrierExtensionAbundances(nn.Module):
    """
    Log-barrier Extension applied to abundances constraints
    See paper arXiv:1904.04205v4 (Kervadec 2020)
    """

    def __init__(self, t):
        """t: parameter of the log-barrier extension."""
        super().__init__()
        self.t = t

    def forward(self, A):
        """
        Return the penalty term for the abundances constraints on A.

        Constraints are: elements between 0 and 1, elements on last
        dimension sum to 1
        """
        pass


def log_barrier_extension(z, t):
    """
    Log-barrier extension function applied on z for parameter t.

    See paper arXiv:1904.04205v4 (Kervadec 2020)
    """
    pass


def smoothing(theta):
    """Compute the pseudo-TV smoothing term for parameters theta."""
    pass
