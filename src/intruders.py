# AI Generated intruder dimension counter based on Shuttleworth et al Algorith 1
"""Intruder dimensions, implementing Shuttleworth et al. (2025), "LoRA vs Full Fine-tuning: An
Illusion of Equivalence", Section 3, page 4 (docs/papers/shuttleworth2025-lora-illusion.pdf).

Definition 3.1: a fine-tuned singular vector y_j is
    "an intruder dimension if and only if max_i(cos(y_j, x_i)) < ϵ"
where x_i are the singular vectors of the pretrained matrix W0.
"""

import torch


# Lines 1 and 2 of Algorithm 1. SVD of W0 and of W_tuned
def leftSingular(W): # Left singular vectors (columns) and singular values of W (float64 for acc)
    U, S, _ = torch.linalg.svd(W.double(), full_matrices=False)  # sorted from largest σ down
    return U, S


# lines 5 and 6 of Algorithm 1.  For top k fine-tuned vectors, cmp with every pretrained vector
def maxSimilarities(U0, Wt, k=10): # For top-k singular vectors of Wt: max |cos| with any col of 
                                    # U0, and σ.
    Ut, St = leftSingular(Wt) # U0: left singular vectors of the pretrained matrix (from
                               # leftSingular), computed once per base.
    sims = (U0.T @ Ut[:, :k]).abs()  # (number of pretrained vectors, k): every |cos| pair
    return sims.max(dim=0).values.tolist(), St[:k].tolist() # Returns two float lists of length k


# Not part of Algorithm 1: the curvature step needs each vector's full rank-one piece σ·u vᵀ, so v too
def topSingularTriplets(W, k=10): # (U[:, :k], S[:k], Vh[:k]) of W in float64, largest σ first
    U, S, Vh = torch.linalg.svd(W.double(), full_matrices=False)  # same call as leftSingular, so u and σ match it
    return U[:, :k], S[:k], Vh[:k]


# Lines 7-10 of Algorithm 1 on saved results: which saved top-k entries are intruders at this ε
def splitByThreshold(entries, eps): # (intruders, others) from dicts with a "maxSim" key, e.g. results/curvature.json entries
    intruders = [entry for entry in entries if entry["maxSim"] < eps]  # strictly below ε, as in Definition 3.1
    others = [entry for entry in entries if entry["maxSim"] >= eps]
    return intruders, others


# Lines 7-10 of Algorithm 1. Count vectors below epsilon
def countIntruders(maxSims, eps):
    return sum(s < eps for s in maxSims) # num of top k vectors whose best match from the
                                         # pretrained vectors < epsilon.
