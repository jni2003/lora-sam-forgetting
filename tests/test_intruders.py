# AI generated planted intruder test for intruder detector
"""Checks for src/intruders.py: plant a known intruder and confirm the detector flags it (brief's check)."""

import torch

from src.intruders import countIntruders, leftSingular, maxSimilarities


# A fake "pretrained" matrix whose directions are stable, so only what we add can create intruders
def baseMatrix(n=256, seed=0): # random n×n matrix with well-separated singular values 10 → 1
    g = torch.Generator().manual_seed(seed)
    U, _ = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=torch.float64)) # random orthonormal directions
    V, _ = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=torch.float64))
    S = torch.diag(torch.linspace(10, 1, n, dtype=torch.float64))               # strengths σ: 10 down to 1
    return U @ S @ V.T                                                           # W0 = U Σ Vᵀ


# False-alarm check: normal small fine-tuning shouldn't create intruders
def test_small_update_has_no_intruders():
    W0 = baseMatrix()
    noise = torch.randn(W0.shape, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    Wt = W0 + 1e-4 * noise                    # tiny update: barely rotates the singular vectors
    sims, _ = maxSimilarities(leftSingular(W0)[0], Wt)
    assert min(sims) > 0.99                   # every top vector still matches a W0 vector (|cos| ≈ 1)
    assert countIntruders(sims, 0.9) == 0


# Why |cos|: -W0 has the same directions with flipped signs, which plain cos would score as -1
def test_sign_flip_is_not_an_intruder():
    W0 = baseMatrix()
    sims, _ = maxSimilarities(leftSingular(W0)[0], -W0)
    assert countIntruders(sims, 0.9) == 0     # fails if the .abs() in maxSimilarities is removed


# The brief's check: plant a strong rank-1 update u vᵀ (like a LoRA update) and confirm it's flagged
def test_planted_rank_one_direction_is_flagged():
    W0 = baseMatrix()
    g = torch.Generator().manual_seed(2)
    u = torch.randn(256, generator=g, dtype=torch.float64)
    v = torch.randn(256, generator=g, dtype=torch.float64)
    u, v = u / u.norm(), v / v.norm()         # random unit directions: |cos| ≈ 0.2 with every W0 vector
    Wt = W0 + 20 * torch.outer(u, v)          # σ 20 beats W0's top σ 10, so u becomes the top direction
    sims, sigmas = maxSimilarities(leftSingular(W0)[0], Wt)
    assert sims[0] < 0.5                      # the top vector is the intruder
    assert 20 < sigmas[0] < 25                # about the planted strength (W0 adds a little along u)
    assert countIntruders(sims, 0.5) == 1     # and it is the only one
