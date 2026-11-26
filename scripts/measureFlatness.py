# AI generated script meaasuring base models' flatness with Lanczos and Hutchinson implementations
"""Whole-model flatness of each base model's pretraining (DCLM) loss over the attention/MLP matrices.

For the SAM and AdamW base models: the most extreme Hessian eigenvalues (Lanczos) and the trace
(Hutchinson), from src/flatness.py. Checks that the SAM model is flatter overall (Foret et al.;
Watts et al.) and gives a scale for the per-intruder κ in results/curvature.json: compare an
intruder's κ with the top eigenvalue (sharpest possible) and with trace / weights (average).

Convergence check: Lanczos is run twice, with fewer and more steps; the top eigenvalue should agree.

Usage (cluster):  python -u -m scripts.measureFlatness --out results/flatness.json
Quick test:       add --rows 1 --lanczos-steps 4 --lanczos-check-steps 3 --hutchinson-samples 2
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.convert import load_olmo_as_llama
from src.curvature import makeChunkedLanguageModelLosses
from src.flatness import hutchinsonTrace, lanczosEigenvalues

MATRIX_TYPES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]  # same matrices as the intruders
EIGENVALUES_TO_SAVE = 5  # top and bottom few; Lanczos only resolves the extremes well

parser = argparse.ArgumentParser()
parser.add_argument("--bases", nargs="+", default=["sam", "adamw"])
parser.add_argument("--base-dir", default="checkpoints", help="folder holding olmo-60m-{sam,adamw}")
parser.add_argument("--data", default="data/dclm_heldout.npy", help="DCLM rows for the pretraining loss")
parser.add_argument("--rows", type=int, default=8, help="rows of 1024 tokens; same batch as the curvature run")
parser.add_argument("--rows-per-chunk", type=int, default=1, help="rows processed at once; results are averaged (exact)")
parser.add_argument("--lanczos-steps", type=int, default=20)
parser.add_argument("--lanczos-check-steps", type=int, default=15, help="fewer steps, to check the top eigenvalue converged")
parser.add_argument("--hutchinson-samples", type=int, default=50)
parser.add_argument("--out", default="results/flatness.json")
args = parser.parse_args()


def attentionAndMlpNames(model): # names of the matrices LoRA adapts, in model order
    names = [name for name, _ in model.named_parameters()
             if any(f".{matrixType}.weight" in name for matrixType in MATRIX_TYPES)]
    return names


def loadBaseModel(base, device): # base model ready for second derivatives: eager attention, float32, on device
    model = load_olmo_as_llama(f"{args.base_dir}/olmo-60m-{base}")
    model.set_attn_implementation("eager")  # fused attention kernels may not support the second backward pass
    return model.to(device)


def measureBase(base, lossChunks, device): # flatness summary for one base model
    model = loadBaseModel(base, device)
    names = attentionAndMlpNames(model)
    numWeights = sum(dict(model.named_parameters())[name].numel() for name in names)

    # Lanczos twice: the smaller run is only there to show the top eigenvalue has converged
    print(f"base {base}: Lanczos ({args.lanczos_steps} and {args.lanczos_check_steps} steps) over {numWeights:,} weights ...")
    eigenvalues = lanczosEigenvalues(lossChunks, model, names, numSteps=args.lanczos_steps)
    checkEigenvalues = lanczosEigenvalues(lossChunks, model, names, numSteps=args.lanczos_check_steps)

    # Hutchinson: average curvature over all directions
    print(f"base {base}: Hutchinson ({args.hutchinson_samples} samples) ...")
    trace, traceStandardError = hutchinsonTrace(lossChunks, model, names, numSamples=args.hutchinson_samples)

    summary = {
        "numWeights": numWeights,
        "topEigenvalues": eigenvalues[:EIGENVALUES_TO_SAVE].tolist(),
        "bottomEigenvalues": eigenvalues[-EIGENVALUES_TO_SAVE:].tolist(),
        "topEigenvalueWithFewerSteps": checkEigenvalues[0].item(),
        "trace": trace,
        "traceStandardError": traceStandardError,
        "averageCurvature": trace / numWeights,  # trace / n: mean of all eigenvalues
    }
    return summary


device = "cuda" if torch.cuda.is_available() else "cpu"
inputIds = torch.from_numpy(np.load(args.data)[: args.rows].astype(np.int64)).to(device)
lossChunks = makeChunkedLanguageModelLosses(inputIds, args.rows_per_chunk)  # pretraining (DCLM) loss: H is its Hessian

startTime = time.time()
results = {"settings": vars(args) | {"device": device}, "bases": {}}
for base in args.bases:
    summary = measureBase(base, lossChunks, device)
    results["bases"][base] = summary
    print(f"base {base}: top eigenvalue {summary['topEigenvalues'][0]:.4g} "
          f"(with fewer steps {summary['topEigenvalueWithFewerSteps']:.4g}), "
          f"most negative {summary['bottomEigenvalues'][-1]:.4g}, "
          f"average curvature {summary['averageCurvature']:.4g} "
          f"(trace {summary['trace']:.4g} ± {summary['traceStandardError']:.2g})")

results["settings"]["minutes"] = (time.time() - startTime) / 60
results["settings"]["peakGpuMemoryGB"] = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None
Path(args.out).write_text(json.dumps(results, indent=1))
print(f"done in {results['settings']['minutes']:.1f} min; saved {args.out}")
