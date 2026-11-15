#!/bin/bash
# AI Generated Script to run commands in the cluster
# Run a command inside the project's Python environment on NYU HPC Cloud Bursting.
#
# On the cluster, Python packages live in a conda environment stored in an "overlay" file (a disk
# image) that is mounted into a Singularity container. Every job runs its Python through this
# wrapper so the job scripts don't repeat the Singularity details.
#
# Set these two once during cluster setup (see the README, "Cluster" section):
OVERLAY="$HOME/lora-env.ext3"                                       # your overlay file with the conda env
IMAGE="/scratch/work/public/singularity/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif"  # CUDA 12 + Ubuntu 22.04 base
#
# Usage (from the repo root, inside a job script):  bash scripts/slurm/run_in_env.sh python -m scripts.evaluate ...

set -euo pipefail  # stop at the first error instead of carrying on

# --nv makes the GPU visible inside the container (harmless on CPU nodes).
# :ro mounts the overlay read-only, so several jobs can use it at the same time.
singularity exec --nv --overlay "$OVERLAY:ro" "$IMAGE" \
    /bin/bash -c "source /ext3/env.sh && cd '$PWD' && $*"
