#!/bin/bash
# AI Generated Slurm job for a single finetune
# GPU job: one fine-tuning run (train -> evaluate -> save). One job per (base, learning rate, method).
#   sbatch scripts/slurm/finetune.sbatch sam 5e-4        # full fine-tuning, SAM base, lr 5e-4
#   sbatch scripts/slurm/finetune.sbatch adamw 5e-4      # same on the AdamW base
#   sbatch scripts/slurm/finetune.sbatch sam 5e-4 8      # LoRA rank 8
# Writes results/<run name>.json (commit it) and checkpoints/finetuned/<run name>/ (weights, not committed).
#SBATCH --job-name=finetune
#SBATCH --account=csci_ga_3033-134-2026fa
#SBATCH --partition=g2-standard-12
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
BASE=$1          # sam or adamw
LR=$2            # peak learning rate, e.g. 5e-4
RANK=${3:-}      # LoRA rank; leave out for full fine-tuning

if [ -z "$RANK" ]; then
    NAME="${BASE}_full_lr${LR}"
    LORA_ARGS=""
else
    NAME="${BASE}_lora${RANK}_lr${LR}"
    LORA_ARGS="--lora-rank $RANK"
fi
echo "run: $NAME"
nvidia-smi

bash scripts/slurm/run_in_env.sh python -m scripts.train \
    --model checkpoints/olmo-60m-$BASE --lr $LR $LORA_ARGS --run-name $NAME \
    --eval-data data/dclm_heldout.npy data/starcoder_val.npy
