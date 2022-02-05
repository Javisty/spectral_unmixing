"""Define the objective function to minimize."""
import torch
import torch.nn as nn
import numpy as np


class SmoothedLinearUnmixing(nn.Module):
    """
    Linear unmixing from the abundances and dispersion model, with smoothed
    regularization terms.
    """

    def __init__(self, target, model, abundances):
        """
        target: spectral image to fit
        model: model used to generate EMs spectra
        abundances: tensor of abundances
        """
        super().__init__()
        self.target = target
        self.model = model
        self.A = abundances

    def forward(self):
        """
        Compute the value of the objective function based on the current
        state.
        """
        pass
