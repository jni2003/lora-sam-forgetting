# AI generated Lanczos and Hutchinson implementations for whole model flatness
"""Whole-model flatness of a loss: the sharpest curvature (Lanczos) and the average curvature (Hutchinson).

These summarize the Hessian H over a set of parameters, unlike src/curvature.py's κ along one
direction. They check that the SAM base model is flatter overall (Foret et al.; Watts et al.)
and give a scale for the per-intruder κ. Both only need Hessian-vector products, so H is never built.

  - Lanczos: after m Hessian-vector products, the eigenvalues of a small m×m tridiagonal matrix
    approximate H's most extreme eigenvalues; the largest is the top curvature λ_max (Yao et al., PyHessian).
  - Hutchinson: E[zᵀHz] = trace(H) for random z with ±1 entries; trace / n is the average curvature.

The parameter set is a list of names, e.g. the 56 attention/MLP matrices (33.6M weights), so each
Lanczos vector costs 134 MB in float32 instead of 545 MB for the whole model with its embeddings.

The loss comes as a list of equal-size row chunks (src/curvature.py makeChunkedLanguageModelLosses);
H·v is averaged over them, which is exact and keeps one chunk's graph in memory. Pass [computeLoss]
for a single loss.
"""

import torch

from src.curvature import hessianVectorProduct

BREAKDOWN_TOLERANCE = 1e-10  # Lanczos stops early if the next vector is ~0: the subspace already holds H's action exactly


def flattenDirection(direction, names): # one long vector, in the order of names
    return torch.cat([direction[name].flatten() for name in names])


def unflattenVector(vector, names, shapes): # inverse of flattenDirection: {name: tensor of that shape}
    chunks = vector.split([shape.numel() for shape in shapes])
    direction = {name: chunk.view(shape) for name, chunk, shape in zip(names, chunks, shapes)}
    return direction


def makeFlatHessianVectorProduct(lossChunks, model, names): # function: flat vector v -> flat H·v over the named parameters
    shapes = [dict(model.named_parameters())[name].shape for name in names]

    def flatHessianVectorProduct(vector):
        direction = unflattenVector(vector, names, shapes)
        # Running sum, not a stacked list: each product is as large as the parameter set
        total = torch.zeros_like(vector)
        for computeLoss in lossChunks:
            total += flattenDirection(hessianVectorProduct(computeLoss, model, direction), names)
        averageProduct = total / len(lossChunks)
        return averageProduct

    return flatHessianVectorProduct


def lanczosEigenvalues(lossChunks, model, names, numSteps=20, seed=0): # approximate extreme Hessian eigenvalues, largest first
    applyHessian = makeFlatHessianVectorProduct(lossChunks, model, names)
    numWeights = sum(dict(model.named_parameters())[name].numel() for name in names)
    firstParameter = next(model.parameters())
    start = torch.randn(numWeights, generator=torch.Generator().manual_seed(seed), dtype=firstParameter.dtype)
    start = start.to(firstParameter.device)
    basis = [start / start.norm()]
    alphas, betas = [], []

    for step in range(numSteps):
        # Apply H to the newest basis vector and record its diagonal entry α in the tridiagonal matrix
        nextVector = applyHessian(basis[-1])
        alpha = torch.dot(nextVector, basis[-1])
        alphas.append(alpha.item())

        # Full reorthogonalization against every basis vector (this also removes the α and β terms of
        # the textbook three-term recurrence); rounding otherwise breaks orthogonality and creates duplicate eigenvalues
        for basisVector in basis:
            nextVector -= torch.dot(nextVector, basisVector) * basisVector

        beta = nextVector.norm().item()
        isLastStep = step == numSteps - 1
        if isLastStep or beta < BREAKDOWN_TOLERANCE:
            break
        betas.append(beta)
        basis.append(nextVector / beta)

    # Eigenvalues of the tridiagonal matrix T (α on the diagonal, β beside it) approximate H's extremes
    T = torch.diag(torch.tensor(alphas, dtype=torch.float64))
    T += torch.diag(torch.tensor(betas, dtype=torch.float64), 1) + torch.diag(torch.tensor(betas, dtype=torch.float64), -1)
    eigenvaluesLargestFirst = torch.linalg.eigvalsh(T).flip(0)
    return eigenvaluesLargestFirst


def hutchinsonTrace(lossChunks, model, names, numSamples=50, seed=0): # (trace estimate, its standard error)
    applyHessian = makeFlatHessianVectorProduct(lossChunks, model, names)
    numWeights = sum(dict(model.named_parameters())[name].numel() for name in names)
    firstParameter = next(model.parameters())
    generator = torch.Generator().manual_seed(seed)

    # Each zᵀHz with ±1 entries is an unbiased estimate of trace(H); average them and report the spread
    samples = []
    for _ in range(numSamples):
        signs = torch.randint(0, 2, (numWeights,), generator=generator) * 2 - 1
        signs = signs.to(device=firstParameter.device, dtype=firstParameter.dtype)
        samples.append(torch.dot(signs, applyHessian(signs)).item())
    samples = torch.tensor(samples, dtype=torch.float64)
    standardError = (samples.std() / numSamples**0.5).item()
    return samples.mean().item(), standardError
