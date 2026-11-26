# AI generated tests for Lanczos and Hutchinson algorithm implementations vs. exact Hessians
"""Checks for src/flatness.py against Hessians whose eigenvalues and trace are known exactly."""

import pytest
import torch

from src.flatness import hutchinsonTrace, lanczosEigenvalues
from tests.test_curvature import Quadratic, exactHessian, mlpInputsAndTargets, mlpLoss, tinyMlp


@pytest.fixture(autouse=True)
def float64():  # float64 so comparisons can be tight; reset afterwards so other test files aren't affected
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(torch.float32)


def allParameterNames(model): # every parameter, so the result can be compared with the full exact Hessian
    return [name for name, _ in model.named_parameters()]


def test_lanczos_with_as_many_steps_as_weights_finds_every_eigenvalue_exactly():
    # With n steps on an n×n Hessian, Lanczos spans the whole space, so T has exactly H's eigenvalues.
    model = Quadratic()
    eigenvalues = lanczosEigenvalues(lambda m: m.loss(), model, ["theta"], numSteps=6)
    expected = torch.linalg.eigvalsh(model.A).flip(0)
    assert torch.allclose(eigenvalues, expected, atol=1e-8)


def test_lanczos_top_eigenvalue_matches_the_exact_hessian_of_a_tiny_mlp():
    # The real use: fewer steps than weights. On this MLP the top eigenvalue's error falls from 4% at
    # 8 steps to 1e-4 at 12 and 1e-8 at 16 (of 26 weights): the extremes converge long before the rest.
    model, (inputs, targets) = tinyMlp(), mlpInputsAndTargets()
    eigenvalues = lanczosEigenvalues(mlpLoss(inputs, targets), model, allParameterNames(model), numSteps=16)
    exactTop = torch.linalg.eigvalsh(exactHessian(model, inputs, targets)).max().item()
    assert abs(eigenvalues[0].item() - exactTop) < 1e-6 * abs(exactTop)


def test_hutchinson_is_exact_for_a_diagonal_hessian():
    # zᵀAz = Σ A_ii z_i² = trace(A) for every ±1 vector z when A is diagonal: no randomness left.
    model = Quadratic()
    model.A = torch.diag(torch.tensor([3.0, -1.0, 0.5, 2.0, 4.0, 1.5]))
    trace, standardError = hutchinsonTrace(lambda m: m.loss(), model, ["theta"], numSamples=5)
    assert abs(trace - 10.0) < 1e-10 and standardError < 1e-10


def test_hutchinson_estimates_the_trace_of_a_tiny_mlp():
    # Unbiased but noisy: with many samples the estimate lands within a few standard errors of the truth.
    model, (inputs, targets) = tinyMlp(), mlpInputsAndTargets()
    trace, standardError = hutchinsonTrace(mlpLoss(inputs, targets), model, allParameterNames(model), numSamples=2000)
    exactTrace = torch.trace(exactHessian(model, inputs, targets)).item()
    assert abs(trace - exactTrace) < 4 * standardError
