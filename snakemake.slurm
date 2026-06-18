#!/bin/bash

# CESGA assigns partition/QOS automatically from requested walltime and memory.
# Leave hardware unconstrained by default so the scheduler can use the broader,
# newer Ice Lake pool instead of restricting jobs to legacy clk nodes.

#SBATCH --job-name=snakemake
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=7-00:00:00
#SBATCH --output=snakemake.out
#SBATCH --error=snakemake.err

# Load environment
module purge
source ~/.bashrc
source ~/.profile
set -euo pipefail
export PATH=/mnt/netapp2/Store_uni/home/bioga/hcx/dcr/juliaup/bin:$PATH
export JULIA_DEPOT_PATH=/mnt/netapp2/Store_uni/home/bioga/hcx/dcr/julia

# Workers must not call CondaPkg resolve() at all — the env is pre-built by
# precompile.slurm. JULIA_CONDAPKG_BACKEND=Null skips resolve() and its
# Pidfile lock entirely. JULIA_CONDAPKG_OFFLINE=true alone is insufficient:
# the lock is acquired before the offline check (CondaPkg.jl issue #153).
# JULIA_PYTHONCALL_EXE points PythonCall directly at the pixi-built Python
# so it never calls envdir(), which raises under the Null backend.
export JULIA_CONDAPKG_BACKEND=Null
export JULIA_PYTHONCALL_EXE=$(pwd)/.CondaPkg/.pixi/envs/default/bin/python

which julia
EXPERIMENT_CONFIG=config/experiment.toml snakemake -s Snakefile.experiment --profile profiles/slurm --rerun-incomplete
