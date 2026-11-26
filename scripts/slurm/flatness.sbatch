#!/bin/bash
# AI generated Slurm job for the flatness script
# Whole-model flatness of both base models (scripts/measureFlatness.py) on one A100.
# Every argument is passed straight to the script, e.g.:
#   sbatch --deadline=now+1hour scripts/slurm/flatness.sbatch --out results/flatness.json
#SBATCH --job-name=flatness
#SBATCH --account=csci_ga_3033-134-2026fa
#SBATCH --partition=c12m85-a100-1  # A100 40GB
#SBATCH --gres=gpu:1
#SBATCH --no-requeue               # ignored for boot failures on this cluster: use sbatch --deadline to cap retries
#SBATCH --time=00:30:00            # includes node boot (up to ~7 min); ~170 Hessian-vector products in total
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
nvidia-smi  # log which GPU we got
bash scripts/slurm/run_in_env.sh python -u -m scripts.measureFlatness "$@"
