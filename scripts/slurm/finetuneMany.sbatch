#!/bin/bash
# AI generated Slurm job running several finetunes in order on one GPU
# Several fine-tuning runs one after another in one GPU job: one node boot per batch instead of one per run.
# Each argument is one run, BASE,LR[,RANK[,SEED]], with the same meaning as finetune.sbatch's arguments, e.g.:
#   sbatch --deadline=now+2hours scripts/slurm/finetuneMany.sbatch sam,2e-4,full,1 adamw,2e-4,full,1 sam,1e-5
# Sequential, not parallel: a run peaks at 13-15 GB, so only two would fit on the A100 and they'd share its compute.
#SBATCH --job-name=finetune-many
#SBATCH --account=csci_ga_3033-134-2026fa
#SBATCH --partition=c12m85-a100-1  # A100 40GB
#SBATCH --gres=gpu:1
#SBATCH --no-requeue               # ignored for boot failures on this cluster: use sbatch --deadline to cap retries
#SBATCH --time=01:15:00            # 4 runs × ~6.5 min + node boot; raise with sbatch --time=... for more runs
#SBATCH --output=slurm-%x-%j.out

set -uo pipefail  # no -e: one failed run shouldn't stop the rest of the batch
for RUN in "$@"; do
    IFS=',' read -r BASE LR RANK SEED <<< "$RUN"  # split "sam,2e-4,full,1" into its four fields
    echo "=== run $RUN"
    bash scripts/slurm/finetune.sbatch "$BASE" "$LR" "${RANK:-}" "${SEED:-}" || echo "!!! run $RUN failed; continuing with the rest"
done
