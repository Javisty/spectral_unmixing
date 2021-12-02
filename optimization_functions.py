"""
Define the elements of optimization problem to solve: objective function
and regularization
"""
import torch
import torch.nn as nn
import numpy as np


class LinearUnmixing(nn.Module):
    """
    Main part: difference between the observed spectra and the linearly
    predicted spectra
    """
    def __init__(self, endmembers, abundances, target):
        super().__init__()
        self.EMs = endmembers
        self.A = abundances
        self.B = target

    def forward(self):
        pass


class AbundanceRegularization(nn.Module):
    """
    Regularization on abundances
    """
    def __init__(self, abundances):
        super().__init__()

    def forward(self):
        pass


class SpectraRegularization(nn.Module):
    """
    Regularization on predicted spectra
    """
    def __init__(self, endmembers):
        super().__init__()

    def forward(self):
        pass
