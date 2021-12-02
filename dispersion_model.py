"""
Encode the dispersion model from Janiczek et al. 2020
"""
import torch
import torch.nn as nn
import numpy as np


class DispersionModel(nn.Module):
    """
    Dispersion model to compute an endmember spectrum
    """
    def __init__(self, wavenumbers):
        super().__init__()

    def predict_spectrum(self, endmember):
        """
        Return the predicted spectrum for the endmember
        """
        pass
