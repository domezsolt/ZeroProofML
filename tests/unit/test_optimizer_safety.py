"""Tests for optimizer safety helper bounds.

Validates the sufficient step-size conditions returned by:
 - eta_heavy_ball
 - eta_adam
and basic behavior of batch-safe curvature proxy.
"""

import math

from zeroproof.optim_utils import BatchCurvatureProxy, batch_safe_lr, eta_adam, eta_heavy_ball


def test_eta_heavy_ball_matches_formula_and_clamps():
    L_hat = 100.0

    # Typical momentum
    beta1 = 0.9
    eta = eta_heavy_ball(BatchCurvatureProxy(L_hat), beta1)
    expected = 2.0 * (1.0 - beta1) / L_hat
    assert math.isclose(eta, expected, rel_tol=1e-12)

    # Higher momentum -> smaller eta
    eta_high_mom = eta_heavy_ball(BatchCurvatureProxy(L_hat), 0.99)
    assert eta_high_mom < eta

    # Clamp beta1 outside [0,1) should still return finite positive eta
    eta_neg = eta_heavy_ball(BatchCurvatureProxy(L_hat), -1.0)
    eta_gt = eta_heavy_ball(BatchCurvatureProxy(L_hat), 2.0)
    assert 0.0 < eta_neg <= 2.0 / L_hat
    assert 0.0 < eta_gt <= 2.0 / L_hat


def test_eta_adam_matches_formula_and_clamps():
    L_hat = 50.0
    beta1 = 0.9
    beta2 = 0.999
    eta = eta_adam(BatchCurvatureProxy(L_hat), beta1, beta2)
    expected = (1.0 - beta1) / (math.sqrt(1.0 - beta2) * L_hat)
    assert math.isclose(eta, expected, rel_tol=1e-12)

    # Stronger momentum -> smaller eta; larger beta2 -> larger eta per bound
    eta_b1 = eta_adam(BatchCurvatureProxy(L_hat), 0.99, beta2)
    eta_b2 = eta_adam(BatchCurvatureProxy(L_hat), beta1, 0.9999)
    assert eta_b1 < eta
    assert eta_b2 > eta

    # Clamp betas outside [0,1) yields finite positive eta
    eta_bad = eta_adam(BatchCurvatureProxy(L_hat), -5.0, 5.0)
    assert eta_bad > 0.0


def test_batch_safe_lr_positive_and_bounded():
    # Use proxy from batch features; returns 1/L_hat scale
    proxy = batch_safe_lr(B_psi=2.0, q_min=0.5, y_max=3.0, alpha=0.0, safety=1.0)
    eta = 1.0 / proxy.L_hat
    assert eta > 0.0
    # Larger y_max increases L_hat -> smaller eta
    proxy2 = batch_safe_lr(B_psi=2.0, q_min=0.5, y_max=5.0, alpha=0.0, safety=1.0)
    assert 1.0 / proxy2.L_hat < 1.0 / proxy.L_hat
