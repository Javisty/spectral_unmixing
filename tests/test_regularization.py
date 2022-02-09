import torch
import numpy as np
import pytest

from src.regularization import (norm_1_2,
                                smoothing,
                                log_barrier_extension,
                                LogBarrierExtensionAbundances)


def test_norm_1_2():
    A = torch.tensor([[1, 0], [1, 0]]).reshape((2, 2, 1, 1))
    B = torch.tensor([[2, 2], [0, 0]]).reshape((2, 2, 1, 1))
    assert norm_1_2(A, B) == pytest.approx(3 + np.sqrt(5))


def test_smoothing():
    theta = torch.from_numpy(np.arange(4).reshape((2, 2, 1, 1)))
    theta = torch.cat((theta, theta, theta), dim=2)
    theta = torch.cat((theta, theta), dim=3)
    assert smoothing(theta) == pytest.approx(3*2*(3 + np.sqrt(5))**2)


class TestLogBarrierExtension():
    def test_left_part(self):
        res = log_barrier_extension(torch.tensor(-0.5), torch.tensor(2))
        assert res == pytest.approx(-np.log(0.5)/2)

    def test_right_part(self):
        expected = -0.2 - np.log(0.25)/2 + 0.5
        res = log_barrier_extension(torch.tensor(-0.1), torch.tensor(2))
        assert res == pytest.approx(expected)

    def test_for_abundances(self):
        A = torch.tensor([[[1/3, 1/3, 1/3], [1, 1, -1], [0.5, 0.5, 1]]])
        res = LogBarrierExtensionAbundances(1).forward(A)
        assert res == pytest.approx((2 + 1 + 1) + (2 + 1 + 1) + (1 + 0 + 2))
