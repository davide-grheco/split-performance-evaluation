#!/bin/bash
#SBATCH --job-name=snakemake_experiment
#SBATCH --output=logs/coordinator_%j.out
#SBATCH --error=logs/coordinator_%j.err
#SBATCH --time=7-00:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

# Run both experiment configs in parallel.
# Each Snakemake coordinator submits its own SLURM jobs and monitors them
# independently; they share the cluster queue but never interfere.
#
# Usage:
#   sbatch submit_experiment.sh
#
# To run only the full or subsampled experiment:
#   sbatch --export=ALL,RUN_FULL=1,RUN_SUB=0 submit_experiment.sh
#   sbatch --export=ALL,RUN_FULL=0,RUN_SUB=1 submit_experiment.sh
module purge
source ~/.bashrc
source ~/.profile
set -euo pipefail

RUN_FULL="${RUN_FULL:-1}"
RUN_SUB="${RUN_SUB:-1}"

# Pick a leaner SLURM profile for subsampled experiments — full-dataset memory
# and runtime ceilings are overkill for 150-sample folds.
PROFILE="profiles/slurm"
if [[ "${EXPERIMENT_CONFIG:-}" == *sub150* || "${EXPERIMENT_CONFIG:-}" == *sub1000* ]]; then
    PROFILE="profiles/slurm_sub150"
fi

snakemake --drop-metadata --rerun-incomplete -s Snakefile.experiment --profile "$PROFILE"
