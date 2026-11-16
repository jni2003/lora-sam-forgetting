# AI gnerated script to count intruders in fine-tuned checkpoints
"""Count intruder dimensions (src/intruders.py) in fine-tuned checkpoints vs. their base model.

Usage (on the cluster, CPU partition):
    python -m scripts.countIntruders --runs checkpoints/finetuned/*_lr5e-4
Writes results/intruders.json (per run, per matrix: maxSims and sigmas).
"""

import argparse
import json
from pathlib import Path

from transformers import LlamaForCausalLM

from src.convert import load_olmo_as_llama
from src.intruders import countIntruders, leftSingular, maxSimilarities

MATRIX_TYPES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]  # same as LoRA targets

parser = argparse.ArgumentParser()
parser.add_argument("--runs", nargs="+", required=True, help="fine-tuned checkpoint folders")
parser.add_argument("--base-dir", default="checkpoints", help="folder holding olmo-60m-{sam,adamw}")
parser.add_argument("--k", type=int, default=10, help="top-k fine-tuned singular vectors to check (paper: 10)")
parser.add_argument("--eps", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
parser.add_argument("--out", default="results/intruders.json")
args = parser.parse_args()


# Pick out the matrices LoRA adapts (q, k, v, o, gate, up, down in every layer): 56 for the 60M model
def attentionAndMlpWeights(model): # {name: weight}, e.g. 'model.layers.3.mlp.up_proj.weight'
    return {name: w.detach() for name, w in model.state_dict().items()
            if any(f".{t}.weight" in name for t in MATRIX_TYPES)}


# For each run: compare every fine-tuned matrix with the same matrix in its base model
baseU = {}  # base name -> {matrix name: U0}; each base's SVDs are done once and reused across runs
results = {"k": args.k, "runs": {}}
for runDir in sorted(args.runs):
    runName = Path(runDir).name
    base = runName.split("_")[0]  # base from the run name's prefix: "sam" or "adamw"

    # Line 1 of Algorithm 1 (SVD of W0), only the first time this base comes up
    if base not in baseU:
        print(f"SVD of base {base} ...")
        baseModel = load_olmo_as_llama(f"{args.base_dir}/olmo-60m-{base}")
        baseU[base] = {n: leftSingular(w)[0] for n, w in attentionAndMlpWeights(baseModel).items()}
        del baseModel

    # Lines 2-6 of Algorithm 1 for every matrix: SVD of W_tuned and best |cos| of its top k vectors
    tuned = attentionAndMlpWeights(LlamaForCausalLM.from_pretrained(runDir))
    assert tuned.keys() == baseU[base].keys(), "fine-tuned and base matrices don't line up"
    matrices = {}
    for name, w in tuned.items():
        sims, sigmas = maxSimilarities(baseU[base][name], w, args.k)
        matrices[name] = {"maxSims": sims, "sigmas": sigmas}
    results["runs"][runName] = {"base": base, "matrices": matrices}

# Print a table: total intruders over all matrices at each ε (out of 56 matrices × k vectors checked)
print(f"\n{'run':28s}" + "".join(f"  ε={e:<4}" for e in args.eps) + "   (of how many checked)")
for runName, run in results["runs"].items():
    allSims = [s for m in run["matrices"].values() for s in m["maxSims"]] # every matrix's top-k scores in one list
    print(f"{runName:28s}" + "".join(f"  {countIntruders(allSims, e):<6d}" for e in args.eps) + f"   {len(allSims)}")

# Save the raw scores (not just counts) so any other ε can be counted later without redoing SVDs
Path(args.out).write_text(json.dumps(results, indent=1))
print(f"saved {args.out}")
