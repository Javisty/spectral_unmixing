"""Define the regularization terms."""
import torch
import torch.nn as nn
from maskedtensor import masked_tensor
import numpy as np


class AbundanceRegularization(nn.Module):
    """Regularization on abundances."""

    def __init__(self,
                 barrier_params=(5, 1.1),
                 barrier_penalty=0.1,
                 spoq_params=(0.25, 2, 7e-7, 3e-3, 0.1),
                 sparsity_penalty=1e-2):
        """
        barrier_params: parameters (t, mu) to define the log-barrier function
        barrier_penalty: penalty on the barrier term
        spoq_params: (p, q, alpha, beta, eta) the tuple of parameters defining
        the pseudo-norm used for sparsity
        sparsity_penalty: penalty on the sparsity term
        """
        super().__init__()
        self.barrier = LogBarrierExtensionAbundances(*barrier_params)
        self.spoq = SPOQ(*spoq_params)
        self.delta = sparsity_penalty
        self.zeta = barrier_penalty

    def forward(self, A):
        """Return the regularization term for abundances A."""
        return self.zeta * self.barrier.forward(A)\
            + self.delta * self.spoq.forward(A)


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

    def __init__(self, t, increase_factor=1.1):
        """
        t: parameter of the log-barrier extension
        increase_factor: factor to multiply t with at each step
        """
        super().__init__()
        self.t = torch.tensor(t)
        self.mu = increase_factor
        na_where = NaNWhere

        @staticmethod
        def forward(ctx, x, f1, f2, mask_fn=self.condition):
            x_1 = x.detach().clone().requires_grad_(True)
            x_2 = x.detach().clone().requires_grad_(True)

            with torch.enable_grad():
                y_1 = f1(x_1)
                y_2 = f2(x_2)

            mask = mask_fn(x)

            ctx.save_for_backward(mask)
            ctx.x_1 = x_1
            ctx.x_2 = x_2
            ctx.y_1 = y_1
            ctx.y_2 = y_2

            return torch.where(mask, y_1, y_2)
        na_where.forward = forward

        self.nan_where = na_where.apply

    def increase_t(self):
        """Change t by a factor self.mu."""
        self.t = self.t * self.mu

    def condition(self, x):
        return x <= -1/self.t**2

    def f1(self, x):
        return -torch.log(-x) / self.t

    def f2(self, x):
        return self.t * x - torch.log(1/self.t**2)/self.t + 1/self.t

    def log_barrier_extension(self, z):
        """Apply log-barrier extension function with parameter self.t."""
        return self.nan_where(z, self.f1, self.f2)

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


class NaNWhere(torch.autograd.Function):
    '''
    An adaptation of torch.where to situations like
    output = torch.where(torch.isfinite(f1(x)), f1(x), f2(x))
    where the point of torch.where is to mask out invalid application
    of f1 to some elements of x and replace them with fallback values
    provided by f2(x).
    '''
    @staticmethod
    def forward(ctx, x, f1, f2, mask_fn=torch.isfinite):
        x_1 = x.detach().clone().requires_grad_(True)
        x_2 = x.detach().clone().requires_grad_(True)

        with torch.enable_grad():
            y_1 = f1(x_1)
            y_2 = f2(x_2)

        mask = mask_fn(x)

        ctx.save_for_backward(mask)
        ctx.x_1 = x_1
        ctx.x_2 = x_2
        ctx.y_1 = y_1
        ctx.y_2 = y_2

        return torch.where(mask, y_1, y_2)

    @staticmethod
    def backward(ctx, gout):
        mask, = ctx.saved_tensors

        torch.autograd.backward([ctx.y_1, ctx.y_2], [gout, gout])
        gin = torch.where(mask, ctx.x_1.grad, ctx.x_2.grad)

        return gin, None, None


def log_barrier_extension(z, t, f1, f2, where_, mask_fn):
    """
    Log-barrier extension function applied on z for parameter t.

    See paper arXiv:1904.04205v4 (Kervadec 2020)
    """
    # mask = (z <= -1/t**2)
    # z_left = masked_tensor(z, mask, requires_grad=True)
    # z_right = masked_tensor(z, ~mask, requires_grad=True)
    # return torch.where(mask,
    #                    -torch.log(-z_left) / t,
    #                    t * z_right - torch.log(1/t**2)/t + 1/t)
    return where_(z, f1, f2)

    # return torch.where(z <= -1/t**2,
    #                    -torch.log(-z) / t,
    #                    t * z - torch.log(1/t**2)/t + 1/t)


def smoothing(theta):
    """Compute the pseudo-TV smoothing term for parameters theta."""
    S, device = theta.size(), theta.device

    h_diff = theta[:, 1:, :, :] - theta[:, :-1, :, :]
    v_diff = theta[1:, :, :, :] - theta[:-1, :, :, :]

    # Pad with zeros to restore dimension
    # h_diff = torch.cat((h_diff, torch.zeros((S[0], 1, S[2], S[3]), device=device)), dim=1)
    # v_diff = torch.cat((v_diff, torch.zeros((1, S[1], S[2], S[3]), device=device)), dim=0)

    sub = norm_1_2(h_diff[:-1, :, :, :], v_diff[:, :-1, :, :], dim=(0, 1))**2
    last_row = torch.sum(h_diff[-1, :, :, :].abs(), dim=0)**2
    last_col = torch.sum(v_diff[:, -1, :, :].abs(), dim=0)**2

    return torch.sum(sub) + torch.sum(last_row) + torch.sum(last_col)


def norm_1_2(A, B, **kwargs):
    """Compute the 1,2-norm for matrices A and B.

    See paper DOI:10.1109/LSP.2014.2322123 (Condat 2014), section III.A
    """
    if A.size() != B.size():
        raise AttributeError("Tensors should have same size")
    return torch.sum(torch.sqrt(A**2 + B**2 + 0.00001), **kwargs)
