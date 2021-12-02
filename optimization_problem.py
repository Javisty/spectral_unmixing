"""
Wrap the optimization problem in one class
"""
import torch
import torch.nn as nn
import numpy as np


class UnmixingProblem:
    """
    Represent the current state of our optimization problem
    """
    def __init__(self, target, endmembers, abundances, wavenumbers, alpha, beta):
        """
        target: spectral image on the wavenumbers
            the target spectra to unmix
        endmembers: a list of Endmember instances
            the initial values for the endmembers parameters
        abundances: tensor of abundances
            the initial values for the abundances
        wavenumbers: array of wavelengths
            the wavelengths used for the spectra
        alpha: float
            regularization weight for the Total Variation
        beta: float
            regularization weight for the abundances
        """
        self.EMs = endmembers
        self.A = abundances
        self.B = target
        self.waves = wavenumbers
        self.alpha = alpha
        self.beta = beta


class Endmember:
    """
    An endmember is characterized by its set of parameters
    """
    def __init__(self, K: int):
        """
        K: int
            number of distinct mass-spring equations
        """
        self.K = K

    def initialize(self):
        """
        Initalize the parameters of the endmember
        """
        pass
