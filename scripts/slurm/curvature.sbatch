#!/bin/bash
# AI generated Slurm job for the curvature script on one A100
# Curvature along each run's top singular directions (scripts/curvatureAlongIntruders.py) on one A100.
# Every argument is passed straight to the script, e.g.:
#   sbatch scripts/slurm/curvature.sbatch --runs checkpoints/finetuned/sam_full_lr8e-4 --max-matrices 2 --out results/curvature_test.json
#SBATCH --job-name=curvature
#SBATCH --account=csci_ga_3033-134-2026fa
#SBATCH --partition=c12m85-a100-1  # A100 40GB: the double backward pass on 8 rows needs ~15-20 GB
#SBATCH --gres=gpu:1
#SBATCH --no-requeue               # a failed node boot ends the job instead of retrying (caps GPU-hours)
#SBATCH --time=01:00:00            # includes node boot (up to ~7 min); override with sbatch --time=...
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
nvidia-smi  # log which GPU we got
bash scripts/slurm/run_in_env.sh python -m scripts.curvatureAlongIntruders "$@"
