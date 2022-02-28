"""Minimize the objective function."""
import torch.nn as nn
from torch.optim import Adam


class AutogradDescent(nn.Module):
    """Do Gradient Descent using autograd functionality."""

    def __init__(self, objective):
        """objective: model to minimize."""
        super().__init__()
        self.obj = objective

    def fit(self, target, epochs=500, learning_rate=1e-4, betas=(0.9, 0.999)):
        """
        Train the model to fit the target Hyperspectral Image, using Adam descent.

        target (torch.Tensor of size (N, M, S)): hyperspectral image to fit
        epochs, learning_rate, betas: parameters for Adam
        """
        self.target = target
        self.optimizer = Adam(self.parameters(), lr=learning_rate, betas=betas)

        for epoch in range(epochs):
            self.step()
            if epoch % 5 == 0:
                print(f"\r Epoch {epoch}\t Current loss: {self.obj.forward()}",
                      end='')

    def step(self):
        """Do one step of Gradient Descent."""
        self.optimizer.zero_grad()
        self.loss = self.forward()
        self.loss.backward()
        self.optimizer.step()
        self.apply_constraints()

    def forward(self):
        """Evaluate the objective at the current state."""
        return self.obj.forward()

    def apply_constraints(self):
        """Apply constraints on parameters."""
        for endmember in self.obj.model.values():
            endmember.apply_constraints()
