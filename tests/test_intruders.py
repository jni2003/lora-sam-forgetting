# AI generated planted intruder test for intruder detector
"""Checks for src/intruders.py: plant a known intruder and confirm the detector flags it (brief's check)."""

import torch

from src.intruders import countIntruders, leftSingular, maxSimilarities, splitByThreshold, topSingularTriplets


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


# The curvature script rebuilds each top vector's rank-one piece σ·u vᵀ from these triplets
def test_top_triplets_match_left_singular_and_rebuild_the_matrix():
    W = baseMatrix(n=32)
    U, S, Vh = topSingularTriplets(W, k=32)
    leftU, leftS = leftSingular(W)
    assert torch.allclose(U, leftU) and torch.allclose(S, leftS)  # same u and σ the intruder counts used
    rebuilt = sum(S[i] * torch.outer(U[:, i], Vh[i]) for i in range(32))  # all rank-one pieces add back to W
    assert torch.allclose(rebuilt, W.double(), atol=1e-10)


# The analysis script relabels saved entries at any ε; an entry exactly at ε is not an intruder (Definition 3.1: "< ϵ")
def test_split_by_threshold_is_strictly_below_eps():
    entries = [{"maxSim": 0.2}, {"maxSim": 0.6}, {"maxSim": 0.9}]
    intruders, others = splitByThreshold(entries, 0.6)
    assert [e["maxSim"] for e in intruders] == [0.2]
    assert [e["maxSim"] for e in others] == [0.6, 0.9]
