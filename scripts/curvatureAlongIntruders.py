# AI generated script to measure curvature along intruder directions with Wats et al eq 3 and 4 implementations
"""Curvature of the pretraining loss along each fine-tuned run's top singular directions.

For every run and every attention/MLP matrix, take the top-k singular vectors (u, σ, v) of the
fine-tuned matrix. Shuttleworth et al.'s Algorithm 1 flags intruders among them by maxSim < ε.
For each one, measure Watts et al.'s κ = dᵀHd / ‖d‖² (Eq. 4) along d = u vᵀ, where H is the
Hessian of the DCLM loss at the base weights. κ doesn't depend on the length of d, so d is unit
length and a vector's share of forgetting is ½ σ² κ (Eq. 3).

All top-k vectors are measured, not only intruders, so any ε can be applied later (as with
results/intruders.json). Baseline: random unit rank-one directions in each matrix of each base.
Spot check: finite differences on a few directions at several step sizes.

Usage (cluster, A100):
    python -m scripts.curvatureAlongIntruders --runs checkpoints/finetuned/sam_full_lr8e-4 \
        checkpoints/finetuned/adamw_full_lr8e-4 --out results/curvature.json
Quick test (any machine): add --max-matrices 2 --rows 1 --random-per-matrix 1
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import LlamaForCausalLM

from src.convert import load_olmo_as_llama
from src.curvature import (directionalCurvature, finiteDifferenceCurvature, makeChunkedLanguageModelLosses,
                           meanOverChunks, rankOneDirection)
from src.intruders import leftSingular, maxSimilarities, topSingularTriplets

MATRIX_TYPES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]  # same as LoRA targets
FINITE_DIFFERENCE_STEPS = [0.3, 0.1, 0.03, 0.01]  # several ε: trust the value where neighbours agree
SUMMARY_EPS = 0.6  # intruder threshold used only for the printed summary; the JSON keeps every maxSim

parser = argparse.ArgumentParser()
parser.add_argument("--runs", nargs="+", required=True, help="fine-tuned checkpoint folders")
parser.add_argument("--base-dir", default="checkpoints", help="folder holding olmo-60m-{sam,adamw}")
parser.add_argument("--data", default="data/dclm_heldout.npy", help="DCLM rows for the pretraining loss")
parser.add_argument("--rows", type=int, default=8, help="rows of 1024 tokens in the DCLM batch")
parser.add_argument("--rows-per-chunk", type=int, default=1, help="rows processed at once; results are averaged (exact), memory scales with this")
parser.add_argument("--k", type=int, default=10, help="top-k fine-tuned singular vectors (Shuttleworth: 10)")
parser.add_argument("--random-per-matrix", type=int, default=3, help="random baseline directions per matrix")
parser.add_argument("--finite-difference-checks", type=int, default=3, help="directions to spot-check")
parser.add_argument("--max-matrices", type=int, default=None, help="only the first N matrices (quick tests)")
parser.add_argument("--seed", type=int, default=0, help="seed for the random baseline directions")
parser.add_argument("--out", default="results/curvature.json")
args = parser.parse_args()


def attentionAndMlpWeights(model): # {name: weight on CPU} for the matrices LoRA adapts
    return {name: weight.detach().cpu() for name, weight in model.state_dict().items()
            if any(f".{matrixType}.weight" in name for matrixType in MATRIX_TYPES)}


def loadBaseModel(base, device): # base model ready for second derivatives: eager attention, float32, on device
    model = load_olmo_as_llama(f"{args.base_dir}/olmo-60m-{base}")
    model.set_attn_implementation("eager")  # fused attention kernels may not support the second backward pass
    return model.to(device)


def chunkedKappa(model, lossChunks, direction): # κ along direction on the whole batch, averaged chunk by chunk (exact)
    kappa = meanOverChunks(lambda computeLoss: directionalCurvature(computeLoss, model, direction)[1], lossChunks)
    return kappa


def randomBaselineKappas(model, lossChunks, matrixNames, generator): # {matrix: [κ along random unit u vᵀ]}
    kappas = {}
    for name in matrixNames:
        numRows, numCols = dict(model.named_parameters())[name].shape
        kappas[name] = []
        for _ in range(args.random_per_matrix):
            u = torch.randn(numRows, generator=generator, dtype=torch.float64)
            v = torch.randn(numCols, generator=generator, dtype=torch.float64)
            direction = rankOneDirection(model, name, u / u.norm(), 1.0, v / v.norm())
            kappas[name].append(chunkedKappa(model, lossChunks, direction))
    return kappas


def topDirectionKappas(model, lossChunks, baseWeights, baseU, tunedWeights, matrixNames): # {matrix: [one entry per top-k vector]}
    matrices = {}
    for name in matrixNames:
        maxSims, _ = maxSimilarities(baseU[name], tunedWeights[name], args.k)
        U, S, Vh = topSingularTriplets(tunedWeights[name], args.k)
        update = (tunedWeights[name] - baseWeights[name]).double()
        entries = []
        for i in range(args.k):
            direction = rankOneDirection(model, name, U[:, i], 1.0, Vh[i])
            kappa = chunkedKappa(model, lossChunks, direction)
            entries.append({
                "sigma": S[i].item(),
                "maxSim": maxSims[i],
                "kappa": kappa,
                # ‖u uᵀ ΔW‖: how much of the update itself lies along u (the alternative intruder piece)
                "updateNormAlongU": (U[:, i] @ update).norm().item(),
            })
        matrices[name] = entries
    return matrices


def finiteDifferenceSpotChecks(model, lossChunks, tunedWeights, matrixNames, runName): # κ from the HVP vs from loss values
    checks = []
    for name in matrixNames[: args.finite_difference_checks]:
        U, _, Vh = topSingularTriplets(tunedWeights[name], 1)
        direction = rankOneDirection(model, name, U[:, 0], 1.0, Vh[0])
        kappa = chunkedKappa(model, lossChunks, direction)
        byStep = {str(eps): meanOverChunks(lambda computeLoss: finiteDifferenceCurvature(computeLoss, model, direction, eps),
                                           lossChunks)
                  for eps in FINITE_DIFFERENCE_STEPS}  # d is unit length, so dᵀHd = κ
        checks.append({"run": runName, "matrix": name, "kappaHvp": kappa, "finiteDifference": byStep})
    return checks


def printSummary(runName, matrices, randomKappas): # mean κ: intruders vs other top vectors vs random
    entries = [entry for matrixEntries in matrices.values() for entry in matrixEntries]
    intruderKappas = [e["kappa"] for e in entries if e["maxSim"] < SUMMARY_EPS]
    otherKappas = [e["kappa"] for e in entries if e["maxSim"] >= SUMMARY_EPS]
    allRandom = [kappa for kappas in randomKappas.values() for kappa in kappas]
    meanOrNone = lambda values: f"{np.mean(values):.4g}" if values else "none"
    print(f"{runName}: intruders (ε={SUMMARY_EPS}) n={len(intruderKappas)} mean κ {meanOrNone(intruderKappas)} | "
          f"other top n={len(otherKappas)} mean κ {meanOrNone(otherKappas)} | random mean κ {meanOrNone(allRandom)}")


device = "cuda" if torch.cuda.is_available() else "cpu"
inputIds = torch.from_numpy(np.load(args.data)[: args.rows].astype(np.int64)).to(device)
lossChunks = makeChunkedLanguageModelLosses(inputIds, args.rows_per_chunk)  # pretraining (DCLM) loss: H is its Hessian
generator = torch.Generator().manual_seed(args.seed)

results = {"settings": {"rows": args.rows, "rowsPerChunk": args.rows_per_chunk, "k": args.k, "data": args.data, "seed": args.seed,
                        "randomPerMatrix": args.random_per_matrix},
           "randomBaselines": {}, "runs": {}, "finiteDifferenceChecks": []}
runsByBase = {}  # base -> run folders, so each base model is loaded once
for runDir in sorted(args.runs):
    runsByBase.setdefault(Path(runDir).name.split("_")[0], []).append(runDir)  # "sam" or "adamw" prefix

startTime = time.time()
for base, runDirs in runsByBase.items():
    # Load the base model once; its matrices and their SVDs are shared by every run on this base
    print(f"base {base}: loading and computing the base SVDs ...")
    model = loadBaseModel(base, device)
    baseWeights = attentionAndMlpWeights(model)
    baseU = {name: leftSingular(weight)[0] for name, weight in baseWeights.items()}
    matrixNames = list(baseWeights)[: args.max_matrices]

    # Random baseline: depends only on the base model, so it's done once per base
    results["randomBaselines"][base] = randomBaselineKappas(model, lossChunks, matrixNames, generator)

    # Each run on this base: κ along every top-k fine-tuned direction, at the base weights
    for runDir in runDirs:
        runName = Path(runDir).name
        print(f"run {runName}: measuring κ along the top-{args.k} directions of {len(matrixNames)} matrices ...")
        tunedWeights = attentionAndMlpWeights(LlamaForCausalLM.from_pretrained(runDir))
        matrices = topDirectionKappas(model, lossChunks, baseWeights, baseU, tunedWeights, matrixNames)
        results["runs"][runName] = {"base": base, "matrices": matrices}
        printSummary(runName, matrices, results["randomBaselines"][base])

        # Spot check on the first run only: it validates the method, not each run
        if not results["finiteDifferenceChecks"]:
            results["finiteDifferenceChecks"] = finiteDifferenceSpotChecks(model, lossChunks, tunedWeights,
                                                                           matrixNames, runName)
    del model

elapsedMinutes = (time.time() - startTime) / 60
peakMemoryGB = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None
results["settings"].update({"device": device, "minutes": elapsedMinutes, "peakGpuMemoryGB": peakMemoryGB})
print(f"done in {elapsedMinutes:.1f} min, peak GPU memory {peakMemoryGB} GB")
for check in results["finiteDifferenceChecks"]:
    print(f"spot check {check['matrix']}: HVP κ {check['kappaHvp']:.4g}, finite differences {check['finiteDifference']}")

Path(args.out).write_text(json.dumps(results, indent=1))
print(f"saved {args.out}")
