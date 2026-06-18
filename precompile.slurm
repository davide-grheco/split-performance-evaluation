#!/bin/bash
# Precompile DataSplitBench and all dependencies on a clk compute node.
#
# Must run on the same architecture as worker jobs so the .ji cache is valid.
# After this completes, worker jobs can use --compiled-modules=existing safely.
#
# Usage:
#   sbatch precompile.slurm

#SBATCH --job-name=julia_precompile
#SBATCH --constraint=clk
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/precompile_%j.out
#SBATCH --error=logs/precompile_%j.err

module purge
source ~/.bashrc
source ~/.profile
set -euo pipefail

export PATH=/mnt/netapp2/Store_uni/home/bioga/hcx/dcr/juliaup/bin:$PATH
export JULIA_DEPOT_PATH=/mnt/netapp2/Store_uni/home/bioga/hcx/dcr/julia

# Parallelise precompilation across the allocated CPUs.
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "Node:       $(hostname)"
echo "Julia:      $(julia --version)"
echo "Depot:      $JULIA_DEPOT_PATH"
echo "Threads:    $JULIA_NUM_THREADS"
echo "Started:    $(date)"

mkdir -p logs

julia --project=. -e '
    using Pkg
    Pkg.instantiate()
    Pkg.precompile(strict=true)
    println("Julia precompilation complete.")
'

# Resolve the CondaPkg environment so workers find it ready.
# Without this every worker races for the same lockfile.
echo "Resolving CondaPkg environment..."
julia --project=. -e '
    import CondaPkg
    CondaPkg.resolve()
    println("CondaPkg resolved: ", CondaPkg.envdir())
'

echo "Finished:   $(date)"
