import torch
import numpy as np
import pytest

from src.regularization import (norm_1_2,
                                smoothing,
                                DispersionRegularization,
                                log_barrier_extension,
                                LogBarrierExtensionAbundances,
                                SPOQ,
                                AbundanceRegularization)


def test_norm_1_2():
    A = torch.tensor([[1, 0], [1, 0]]).reshape((2, 2, 1, 1))
    B = torch.tensor([[2, 2], [0, 0]]).reshape((2, 2, 1, 1))
    assert norm_1_2(A, B) == pytest.approx(3 + np.sqrt(5))


def test_smoothing():
    theta = torch.from_numpy(np.arange(4).reshape((2, 2, 1, 1)))
    theta = torch.cat((theta, theta, theta), dim=2)
    theta = torch.cat((theta, theta), dim=3)
    assert smoothing(theta) == pytest.approx(3*2*(3 + np.sqrt(5))**2)


def test_dispersion_regu():
    theta = torch.from_numpy(np.arange(4).reshape((2, 2, 1, 1)))
    theta = torch.cat((theta, theta, theta), dim=2)
    theta = torch.cat((theta, theta), dim=3)
    regu = DispersionRegularization(0.1)
    assert regu(theta) == pytest.approx(0.1*3*2*(3 + np.sqrt(5))**2)


class TestLogBarrierExtension:
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


class TestSPOQ:
    def test_l_p_alpha(self):
        spoq = SPOQ(1/4, 2, 0.0000007, 0.003, 0.1)
        A = torch.zeros((2, 1, 5))
        res = spoq.l_p_alpha(A)
        assert res == pytest.approx(0.) and res.dim() == 2

    def test_l_q_eta(self):
        spoq = SPOQ(1/4, 2, 0.0000007, 0.003, 0.1)
        A = torch.zeros((2, 1, 5))
        res = spoq.l_q_eta(A)
        assert res == pytest.approx(spoq.eta) and res.dim() == 2

    def test_spoq(self):
        spoq = SPOQ(1/4, 2, 0.0000007, 0.003, 0.1)
        A = torch.zeros((2, 1, 5))
        expected = 2 * 1 * np.log(spoq.beta / spoq.eta)
        assert spoq.forward(A) == pytest.approx(expected)


def test_abundance_regu():
    a_reg = AbundanceRegularization()
    A = torch.zeros((2, 1, 5))
    expected1 = 0.01 * 2 * 1 * np.log(a_reg.spoq.beta / a_reg.spoq.eta)
    expected2 = 10 * (np.log(25)/5 + 0.2) + 2 * (0 + 5 + np.log(25)/5 + 0.2)
    assert a_reg.forward(A) == pytest.approx(expected1 + expected2)
