#!/bin/bash
# AI generated Slurm job to evaluate DCLM and StarCoder loss of both base models (pre-finetuning baseline)
# GPU job: evaluate both base models (the "before fine-tuning" numbers that forgetting is measured from).
#   sbatch scripts/slurm/eval_base.sbatch        (after prepare.sbatch has finished)
#SBATCH --job-name=eval_base
#SBATCH --account=csci_ga_3033-134-2026fa
#SBATCH --partition=g2-standard-12
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
RUN="bash scripts/slurm/run_in_env.sh"
nvidia-smi  # log which GPU we got

for BASE in sam adamw; do
    $RUN python -m scripts.evaluate --model checkpoints/olmo-60m-$BASE \
        --data data/dclm_heldout.npy data/starcoder_val.npy --out results/base_eval_$BASE.json
done
