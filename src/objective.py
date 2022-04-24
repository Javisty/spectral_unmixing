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
        self.predicted = self.model.forward()
        self.A = nn.Parameter(abundances)
        self.N, self.M, self.P = abundances.size()  # dimensions of problem
        self.regu_A = regu_ab
        self.regu_E = regu_model

    def reconstruct(self):
        """Return the reconstructed spectrum at the current state."""
        abundances = self.A.reshape(self.N, self.M, self.P, 1)
        return torch.sum(self.predicted * abundances, dim=2)

    def reconstruction_error(self):
        """
        Compute the reconstruction error of the image based on the current
        state.
        """
        return torch.sum((self.target - self.reconstruct())**2) / (self.N * self.M)

    def regularization_penalty(self):
        """Compute the regularization penalty based on the current state."""
        return self.regu_A.forward(self.A) + self.regu_E.forward(self.predicted)

    def forward(self):
        """
        Compute the reconstruction error of the image based on the current
        state.
        """
        self.predicted = self.model.forward()
        return self.reconstruction_error() + self.regularization_penalty()
