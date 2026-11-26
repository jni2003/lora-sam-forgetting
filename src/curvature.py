# AI generated file for getting Hessian vector product and curvature along a direction with Watts et al.'s Equations 3 and 4'
"""Curvature of a loss along a direction in weight space, using Hessian-vector products.

Watts et al. (Eq. 3-4): forgetting ≈ ½ ΔᵀHΔ = ½ ‖Δ‖² κ(Δ; H), where H is the Hessian of the
pretraining loss at the base weights and κ(d; H) = dᵀHd / ‖d‖² is the curvature along d.

H has one row and column per weight (~10^8 x 10^8), so it is never built: H·d comes from two
backward passes (Pearlmutter's trick).

A direction is a dict {parameter name: tensor with that parameter's shape}. Parameters missing
from the dict are zero in the direction and are skipped, which is exact for dᵀHd.

For real models, load with attn_implementation="eager": fused attention kernels may not support
the second backward pass.
"""

import torch
import torch.nn.functional as F


def innerProduct(directionA, directionB):  # sum over all parameters of the elementwise product
    return sum((directionA[name] * directionB[name]).sum() for name in directionA)


def hessianVectorProduct(computeLoss, model, direction):  # H·d for the loss computeLoss(model), only on the direction's parameters
    parameters = dict(model.named_parameters())
    names = list(direction)
    weights = [parameters[name] for name in names]

    # First backward pass: the gradient g, keeping its graph so we can differentiate it again
    gradients = torch.autograd.grad(computeLoss(model), weights, create_graph=True)
    gradientDotDirection = sum((gradient * direction[name]).sum() for gradient, name in zip(gradients, names))

    # Second backward pass: the gradient of gᵀd with respect to the weights is H·d
    hessianTimesDirection = torch.autograd.grad(gradientDotDirection, weights)
    result = {name: product.detach() for name, product in zip(names, hessianTimesDirection)}
    return result


def directionalCurvature(computeLoss, model, direction):  # (dᵀHd, κ = dᵀHd / ‖d‖²), Watts et al. Eq. 4
    hessianTimesDirection = hessianVectorProduct(computeLoss, model, direction)
    dHd = innerProduct(direction, hessianTimesDirection).item()
    kappa = dHd / innerProduct(direction, direction).item()
    return dHd, kappa


@torch.no_grad()
def finiteDifferenceCurvature(computeLoss, model, direction, eps=1e-3):  # dᵀHd ≈ [L(θ+εd) + L(θ-εd) - 2L(θ)] / ε², from loss values only
    # An independent check of the HVP. Pick eps by trying a few values: too small and rounding
    # noise (divided by ε²) dominates, too large and the quadratic approximation breaks.
    parameters = dict(model.named_parameters())
    originalWeights = {name: parameters[name].clone() for name in direction}

    def lossAfterStep(stepSize):  # loss at θ + stepSize·d
        for name, directionPart in direction.items():
            parameters[name].copy_(originalWeights[name] + stepSize * directionPart)
        return computeLoss(model).item()

    curvature = (lossAfterStep(eps) + lossAfterStep(-eps) - 2 * lossAfterStep(0.0)) / eps**2

    # Restore the exact original weights so the check has no side effects
    for name in direction:
        parameters[name].copy_(originalWeights[name])
    return curvature


def rankOneDirection(model, name, u, sigma, v):  # σ·u vᵀ in one weight matrix, zero elsewhere: one intruder's piece
    weight = dict(model.named_parameters())[name]
    intruderPiece = (sigma * torch.outer(u, v)).to(dtype=weight.dtype, device=weight.device)
    return {name: intruderPiece}


def makeLanguageModelLoss(inputIds):  # function: model -> mean next-token loss on fixed token rows
    # Computed here, not with model(labels=...).loss, because transformers casts the scores to
    # float32 there, which hides small loss changes in float64 checks.
    def computeLoss(model):
        logits = model(input_ids=inputIds).logits  # (batch, seqLen, vocab)
        # Position t predicts token t+1, as in src/evaluate.py
        loss = F.cross_entropy(logits[:, :-1].flatten(0, 1), inputIds[:, 1:].flatten())
        return loss

    return computeLoss
