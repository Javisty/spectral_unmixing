"""Minimize the objective function."""
import torch
import torch.nn as nn
import numpy as np

class AutogradDescent(nn.Module):
    """Do Gradient Descent using autograd functionality."""

    def __init__(self, objective):
        """objective: model to minimize."""
        super().__init__()
        self.obj = objective

    def forward(self):
        """Evaluate the objective at the current state."""
        pass

    def run(self):
        """Launch the descent process."""
