"""Define the objective function to minimize."""
import torch
import torch.nn as nn


class SmoothedLinearUnmixing(nn.Module):
    """
    Linear unmixing from the abundances and dispersion model, with smoothed
    regularization terms.
    """

    def __init__(self, target, model, abundances, regu_ab, regu_model):
        """
        target: spectral image to fit
        model: model used to generate EMs spectra
        abundances: tensor of abundances
        regu_ab: regularization model for abundances
        regu_model: regularization model for model
        """
        super().__init__()
        self.target = target
        self.model = model
        self.A = abundances
        self.N, self.M, self.P = abundances.size()  # dimensions of problem
        self.regu_A = regu_ab
        self.regu_E = regu_model

    def reconstruction_error(self):
        """
        Compute the reconstruction error of the image based on the current
        state.
        """
        abundances = self.A.reshape(self.N, self.M, self.P, 1)
        reconstruction = torch.sum(self.model.forward() * abundances, dim=2)
        return torch.sum((self.target - reconstruction)**2)

    def regularization_penalty(self):
        """Compute the regularization penalty based on the current state."""
        return self.regu_A.forward(self.A) + self.regu_E.forward()

    def forward(self):
        """
        Compute the reconstruction error of the image based on the current
        state.
        """
        return self.reconstruction_error() + self.regularization_penalty()
