"""Define the regularization terms."""
import torch
import torch.nn as nn
import numpy as np


class AbundanceRegularization(nn.Module):
    """Regularization on abundances."""

    def __init__(self,
                 t_barrier=5,
                 spoq_params=(0.25, 2, 7e-7, 3e-3, 0.1),
                 sparsity_penalty=1e-2):
        """
        t_barrier: parameter to define the log-barrier function.
        spoq_params: (p, q, alpha, beta, eta) the tuple of parameters defining
        the pseudo-norm used for sparsity
        sparsity_penalty: penalty on the sparsity term
        """
        super().__init__()
        self.barrier = LogBarrierExtensionAbundances(t_barrier)
        self.spoq = SPOQ(*spoq_params)
        self.delta = sparsity_penalty

    def forward(self, A):
        """Return the regularization term for abundances A."""
        return self.barrier.forward(A) + self.delta * self.spoq.forward(A)


class DispersionRegularization(nn.Module):
    """Regularization on Dispersion model parameters."""

    def __init__(self, penalty):
        """penalty: the penalty on the regularization term."""
        super().__init__()
        self.penalty = penalty

    def forward(self, theta):
        """Return the regularization term for parameters theta."""
        return self.penalty * smoothing(theta)


class SPOQ(nn.Module):
    """
    Pseudo-norm SPOQ.

    See paper DOI:10.1109/TSP.2020.3025731 (Cherni 2020)
    """

    def __init__(self, p, q, alpha, beta, eta):
        """Parameters for the pseudo-norm."""
        super().__init__()
        self.p = p
        self.q = q
        self.alpha = alpha
        self.beta = beta
        self.eta = eta

    def l_p_alpha(self, A):
        """First quasi-norm approx to the power p."""
        p, alph = self.p, self.alpha
        return torch.sum((A**2 + alph**2)**(p/2) - alph**p, dim=-1)

    def l_q_eta(self, A):
        """Second quasi-norm approx."""
        q, eta = self.q, self.eta
        return (eta**q + torch.sum(torch.abs(A)**q, dim=-1))**(1/q)

    def forward(self, A):
        """Return the SPOQ pseudo-norm for A."""
        p, beta = self.p, self.beta
        l_p_a = self.l_p_alpha(A)
        l_q_n = self.l_q_eta(A)
        return torch.sum(torch.log(((l_p_a + beta**p)**(1/p))/l_q_n))


class LogBarrierExtensionAbundances(nn.Module):
    """
    Log-barrier Extension applied to abundances constraints.

    See paper arXiv:1904.04205v4 (Kervadec 2020)
    """

    def __init__(self, t):
        """t: parameter of the log-barrier extension."""
        super().__init__()
        self.t = torch.tensor(t)

    def log_barrier_extension(self, z):
        """Apply log-barrier extension function with parameter self.t."""
        return log_barrier_extension(z, self.t)

    def forward(self, A):
        """
        Return the penalty term for the abundances constraints on A.

        Constraints are: elements between 0 and 1, elements on last
        dimension sum to 1
        """
        # Penalty term for positive coefficients
        positive = torch.sum(self.log_barrier_extension(-A))

        # Penalty term for sum >= 1 over last axis
        gt_1 = torch.sum(self.log_barrier_extension(torch.sum(A, dim=-1) - 1))
        # Penalty term for sum <= 1 over last axis
        lt_1 = torch.sum(self.log_barrier_extension(1 - torch.sum(A, dim=-1)))

        return positive + lt_1 + gt_1


def log_barrier_extension(z, t):
    """
    Log-barrier extension function applied on z for parameter t.

    See paper arXiv:1904.04205v4 (Kervadec 2020)
    """
    return torch.where(z <= -1/t**2,
                       -torch.log(-z) / t,
                       t * z - torch.log(1/t**2)/t + 1/t)


def smoothing(theta):
    """Compute the pseudo-TV smoothing term for parameters theta."""
    S = theta.size()

    h_diff = theta[:, 1:, :, :] - theta[:, :-1, :, :]
    v_diff = theta[1:, :, :, :] - theta[:-1, :, :, :]

    # Pad with zeros to restore dimension
    h_diff = torch.cat((h_diff, torch.zeros((S[0], 1, S[2], S[3]))), dim=1)
    v_diff = torch.cat((v_diff, torch.zeros((1, S[1], S[2], S[3]))), dim=0)

    return torch.sum(norm_1_2(h_diff, v_diff, dim=(0, 1))**2)


def norm_1_2(A, B, **kwargs):
    """Compute the 1,2-norm for matrices A and B.

    See paper DOI:10.1109/LSP.2014.2322123 (Condat 2014), section III.A
    """
    if A.size() != B.size():
        raise AttributeError("Tensors should have same size")
    return torch.sum(torch.sqrt(A**2 + B**2), **kwargs)
