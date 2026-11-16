#!/bin/bash
# AI Generated Slurm job for a single finetune
# GPU job: one fine-tuning run (train -> evaluate -> save). One job per (base, learning rate, method).
#   sbatch scripts/slurm/finetune.sbatch sam 5e-4        # full fine-tuning, SAM base, lr 5e-4
#   sbatch scripts/slurm/finetune.sbatch adamw 5e-4      # same on the AdamW base
#   sbatch scripts/slurm/finetune.sbatch sam 5e-4 8      # LoRA rank 8
#   sbatch scripts/slurm/finetune.sbatch sam 5e-4 8 1    # LoRA rank 8, seed 1 (run name ..._seed1)
# Writes results/<run name>.json (commit it) and checkpoints/finetuned/<run name>/ (weights, not committed).
#SBATCH --job-name=finetune
#SBATCH --account=csci_ga_3033-134-2026fa
#SBATCH --partition=c12m85-a100-1  # A100 40GB (L4 nodes failed to boot on 2026-11-14)
#SBATCH --gres=gpu:1
#SBATCH --no-requeue           # a failed node boot ends the job instead of retrying (caps GPU-hours)
#SBATCH --time=00:30:00        # includes node boot (up to ~7 min); a run is ~6.5 min
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
BASE=$1          # sam or adamw
LR=$2            # peak learning rate, e.g. 5e-4
RANK=${3:-}      # LoRA rank; use "full" or leave out for full fine-tuning
SEED=${4:-}      # optional seed; leave out for the default (6198). A seed is added to the run name so it doesn't overwrite

if [ -z "$RANK" ] || [ "$RANK" = "full" ]; then
    NAME="${BASE}_full_lr${LR}"
    LORA_ARGS=""
else
    NAME="${BASE}_lora${RANK}_lr${LR}"
    LORA_ARGS="--lora-rank $RANK"
fi
SEED_ARGS=""
if [ -n "$SEED" ]; then
    NAME="${NAME}_seed${SEED}"
    SEED_ARGS="--seed $SEED"
fi
echo "run: $NAME"
nvidia-smi

bash scripts/slurm/run_in_env.sh python -m scripts.train \
    --model checkpoints/olmo-60m-$BASE --lr $LR $LORA_ARGS $SEED_ARGS --run-name $NAME \
    --eval-data data/dclm_heldout.npy data/starcoder_val.npy
