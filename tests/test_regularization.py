import torch
import numpy as np
import pytest

from src.regularization import norm_1_2, smoothing


def test_norm_1_2():
    A = torch.tensor([[1, 0], [1, 0]]).reshape((2, 2, 1, 1))
    B = torch.tensor([[2, 2], [0, 0]]).reshape((2, 2, 1, 1))
    assert norm_1_2(A, B) == pytest.approx(3 + np.sqrt(5))


def test_smoothing():
    theta = torch.from_numpy(np.arange(4).reshape((2, 2, 1, 1)))
    theta = torch.cat((theta, theta, theta), dim=2)
    theta = torch.cat((theta, theta), dim=3)
    assert smoothing(theta) == pytest.approx(3*2*(3 + np.sqrt(5))**2)
