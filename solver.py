"""
Solve the optimization problem with Proximal Alternating Linearized Minimization
"""
import torch
import torch.nn as nn
import numpy as np

from optimization_problem import UnmixingProblem


class PALM(nn.Module):
    """
    Implement the PALM algorithm for our optimization problem
    """
    def __init__(self, init, gamma1: float, gamma2: float):
        """
        init: UnmixingProblem instance
            the initial state
        gamma1, gamma2: float
            update parameters for PALM
        """
        super().__init__()
        self.problem = init
        self.g1, self.g2 = gamma1, gamma2

    def minimize(self):
        """
        Run the algorithm
        """
        pass

    def get_proximal_abundance(self):
        """
        Return the proximal choice for abundances
        """
        pass

    def get_proximal_endmembers(self):
        """
        Return the proximal chocie for endmember parameters
        """
        pass
