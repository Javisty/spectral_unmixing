"""Minimize the objective function."""
import torch.nn as nn
from torch.optim import Adam


class AutogradDescent(nn.Module):
    """Do Gradient Descent using autograd functionality."""

    def __init__(self, objective):
        """objective: model to minimize."""
        super().__init__()
        self.obj = objective
        self.memory = dict({'RE': [], 'barrier': [], 'sparsity': [],
                            'loss': []})

    def fit(self, target, epochs=500, learning_rate=1e-4, betas=(0.9, 0.999)):
        """
        Train the model to fit the target Hyperspectral Image, using Adam descent.

        target (torch.Tensor of size (N, M, S)): hyperspectral image to fit
        epochs, learning_rate, betas: parameters for Adam
        """
        self.target = target
        self.optimizer = Adam(self.parameters(), lr=learning_rate, betas=betas)
        self.save_state()

        for epoch in range(epochs):
            self.step()
            self.save_state()
            if epoch % 5 == 0:
                print(f"\r Epoch {epoch}\t Current loss: {self.obj.forward()}",
                      end='')

    def step(self):
        """Do one step of Gradient Descent."""
        self.optimizer.zero_grad()
        self.loss = self.forward()
        self.loss.backward(retain_graph=False)
        self.optimizer.step()
        self.apply_constraints()
        self.obj.regu_A.barrier.increase_t()

    def forward(self):
        """Evaluate the objective at the current state."""
        return self.obj.forward()

    def save_state(self):
        """Save various values during optimization."""
        memory = self.memory
        memory['loss'].append(float(self.obj.forward()))
        memory['RE'].append(float(self.obj.reconstruction_error()))
        A = self.obj.A
        memory['barrier'].append(float(self.obj.regu_A.barrier.forward(A)))
        memory['sparsity'].append(float(self.obj.regu_A.spoq.forward(A)))

    def apply_constraints(self):
        """Apply constraints on parameters."""
        for endmember in self.obj.model.values():
            endmember.apply_constraints()
