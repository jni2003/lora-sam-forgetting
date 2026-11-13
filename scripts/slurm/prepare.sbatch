#!/bin/bash
# AI Generated Slurm job to download checkpoints and build data
# CPU job: download the two 60M checkpoints and build all data (no GPU needed, so no GPU-hours).
#   sbatch scripts/slurm/prepare.sbatch          (from the repo root)
# Needs `hf auth login` done once, and the StarCoder terms accepted on that account.
#SBATCH --job-name=prepare
#SBATCH --account=csci_ga_3033-134-2026fa
#SBATCH --partition=n2c48m24
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
RUN="bash scripts/slurm/run_in_env.sh"

# Only config.yaml and model.pt; never optim.pt (over 1 GB, unused).
$RUN hf download WattsIshaan/OLMo-60m-192B-sam-cosine config.yaml model.pt --local-dir checkpoints/olmo-60m-sam
$RUN hf download WattsIshaan/OLMo-60m-192B-adamw-cosine config.yaml model.pt --local-dir checkpoints/olmo-60m-adamw

$RUN python -m scripts.make_dclm_heldout   # 20M tokens, 100 files -> data/dclm_heldout.npy (+ _fileids.npy)
$RUN python -m scripts.make_starcoder      # 10M train + 1M val -> data/starcoder_{train,val}.npy
ls -lh data checkpoints/*
