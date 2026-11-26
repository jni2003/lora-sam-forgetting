# AI generated test file for testing the code that checks curvature against the exact Hessian, a pure quadratic, and finite difference
"""Checks for src/curvature.py against cases where the answer is known exactly."""

import pytest
import torch
from torch.func import functional_call
from transformers import LlamaConfig, LlamaForCausalLM

from src.curvature import (directionalCurvature, finiteDifferenceCurvature, hessianVectorProduct, intruderForgettingShare,
                           innerProduct, makeLanguageModelLoss, rankOneDirection)


@pytest.fixture(autouse=True)
def float64():  # float64 so comparisons can be tight; reset afterwards so other test files aren't affected
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(torch.float32)


def tinyMlp():  # 26 weights: small enough for the exact Hessian; tanh so the Hessian isn't constant
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 2))


def mlpInputsAndTargets():  # 5 random examples for the tiny MLP
    generator = torch.Generator().manual_seed(1)
    inputs = torch.randn(5, 3, generator=generator)
    targets = torch.randn(5, 2, generator=generator)
    return inputs, targets


def mlpLoss(inputs, targets):  # mean squared error as a function of the model
    return lambda model: ((model(inputs) - targets) ** 2).mean()


def randomDirection(model, seed=2):  # a random direction over every parameter
    generator = torch.Generator().manual_seed(seed)
    return {name: torch.randn(parameter.shape, generator=generator) for name, parameter in model.named_parameters()}


def flattenDirection(direction, model):  # one long vector, in named_parameters() order
    return torch.cat([direction[name].flatten() for name, _ in model.named_parameters()])


def exactHessian(model, inputs, targets):  # PyTorch's full Hessian over all parameters, flattened
    names = [name for name, _ in model.named_parameters()]
    shapes = [parameter.shape for _, parameter in model.named_parameters()]
    flatWeights = torch.cat([parameter.detach().flatten() for _, parameter in model.named_parameters()])

    def lossOfFlatWeights(flat):  # same loss as mlpLoss, but taking all weights as one vector
        chunks = flat.split([shape.numel() for shape in shapes])
        weights = {name: chunk.view(shape) for name, chunk, shape in zip(names, chunks, shapes)}
        return ((functional_call(model, weights, (inputs,)) - targets) ** 2).mean()

    return torch.autograd.functional.hessian(lossOfFlatWeights, flatWeights)


def test_hvp_matches_the_exact_hessian():
    model, (inputs, targets) = tinyMlp(), mlpInputsAndTargets()
    direction = randomDirection(model)
    H = exactHessian(model, inputs, targets)
    hessianTimesDirection = hessianVectorProduct(mlpLoss(inputs, targets), model, direction)
    expected = H @ flattenDirection(direction, model)
    assert torch.allclose(flattenDirection(hessianTimesDirection, model), expected, atol=1e-10)


def test_direction_in_one_matrix_matches_that_block_of_the_full_hessian():
    # Skipping parameters outside the direction must not change dᵀHd, since they're zero in d.
    model, (inputs, targets) = tinyMlp(), mlpInputsAndTargets()
    computeLoss = mlpLoss(inputs, targets)
    lastWeightPart = randomDirection(model)["2.weight"]
    zeroPadded = {name: torch.zeros_like(parameter) for name, parameter in model.named_parameters()}
    zeroPadded["2.weight"] = lastWeightPart
    dHdWithZeros, _ = directionalCurvature(computeLoss, model, zeroPadded)
    dHdSkipped, _ = directionalCurvature(computeLoss, model, {"2.weight": lastWeightPart})
    assert abs(dHdWithZeros - dHdSkipped) < 1e-10


class Quadratic(torch.nn.Module):  # L(θ) = ½ θᵀAθ + bᵀθ, whose Hessian is exactly A everywhere
    def __init__(self, size=6):
        super().__init__()
        generator = torch.Generator().manual_seed(3)
        M = torch.randn(size, size, generator=generator)
        self.A = M @ M.T  # M Mᵀ is symmetric, as a Hessian must be
        self.b = torch.randn(size, generator=generator)
        self.theta = torch.nn.Parameter(torch.randn(size, generator=generator))

    def loss(self):
        return 0.5 * self.theta @ self.A @ self.theta + self.b @ self.theta


def test_quadratic_loss_change_equals_half_dHd_exactly():
    # Watts et al.'s Eq. 3 has no approximation error for a quadratic: L(θ+d) - L(θ) - gᵀd = ½ dᵀHd.
    model = Quadratic()
    step = torch.randn(6, generator=torch.Generator().manual_seed(4))
    dHd, kappa = directionalCurvature(lambda m: m.loss(), model, {"theta": step})
    assert abs(dHd - (step @ model.A @ step).item()) < 1e-9
    assert abs(kappa - dHd / (step @ step).item()) < 1e-12

    # Take the step for real and compare the actual loss change with the formula
    with torch.no_grad():
        lossBefore = model.loss().item()
        gradientDotStep = ((model.A @ model.theta + model.b) @ step).item()
        model.theta += step
        lossChange = model.loss().item() - lossBefore
    assert abs(lossChange - gradientDotStep - 0.5 * dHd) < 1e-9


def test_finite_difference_agrees_and_restores_the_weights():
    model, (inputs, targets) = tinyMlp(), mlpInputsAndTargets()
    computeLoss = mlpLoss(inputs, targets)
    weightsBefore = {name: parameter.clone() for name, parameter in model.named_parameters()}
    direction = randomDirection(model)
    dHd, _ = directionalCurvature(computeLoss, model, direction)
    assert abs(finiteDifferenceCurvature(computeLoss, model, direction, eps=1e-4) - dHd) < 1e-5 * abs(dHd)
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, weightsBefore[name])  # the check must leave the model exactly as it was


def test_rank_one_intruder_direction_on_a_tiny_llama():
    # The real use: curvature of a language model's loss along u σ vᵀ in one weight matrix.
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                         num_attention_heads=2, max_position_embeddings=32, tie_word_embeddings=False,
                         initializer_range=0.3, attn_implementation="eager")  # larger init: a freshly initialized tiny model is nearly flat
    model = LlamaForCausalLM(config).double().eval()  # transformers builds float32 by default
    inputIds = torch.randint(0, 64, (3, 12), generator=torch.Generator().manual_seed(5))
    computeLoss = makeLanguageModelLoss(inputIds)

    # An MLP output matrix feeds the residual stream directly, so its curvature is clearly nonzero
    name = "model.layers.1.mlp.down_proj.weight"  # shape (16, 32)
    randomMatrix = torch.randn(16, 32, generator=torch.Generator().manual_seed(6))
    U, _, Vh = torch.linalg.svd(randomMatrix, full_matrices=False)
    direction = rankOneDirection(model, name, U[:, 0], 5.0, Vh[0])
    assert list(direction) == [name] and torch.linalg.matrix_rank(direction[name]) == 1
    assert innerProduct(direction, direction).item() > 0

    # eps and tolerance are looser than for the MLP: transformers' attention computes its softmax in
    # float32, so each loss value carries rounding noise, which the finite difference divides by ε².
    dHd, _ = directionalCurvature(computeLoss, model, direction)
    assert abs(finiteDifferenceCurvature(computeLoss, model, direction, eps=3e-2) - dHd) < 1e-2 * abs(dHd)


def test_intruder_forgetting_share_uses_update_size_not_sigma():
    # σ and the update's size along u differ on purpose: the share must use the update's size.
    intruders = [{"sigma": 10.0, "updateNormAlongU": 2.0, "kappa": 0.5},
                 {"sigma": 4.0, "updateNormAlongU": 3.0, "kappa": -0.1}]  # κ can be negative
    expected = 0.5 * 2.0**2 * 0.5 + 0.5 * 3.0**2 * (-0.1)  # = 1.0 - 0.45 = 0.55, worked by hand
    assert abs(intruderForgettingShare(intruders) - expected) < 1e-12
    assert intruderForgettingShare([]) == 0  # no intruders, no share
