#!/bin/bash

set -euo pipefail

# Directory containing Merck CSVs
DATA_DIR="${DATA_DIR:-Data/merck}"
# Use new splitter names matching get_splitter in run_split.jl
SPLITTERS=("butina" "kennardstone" "random" "stratified")

# CESGA assigns the execution queue automatically from requested time/memory.
# Keep these split jobs within the 6h default path, but do not request a
# partition, QOS, or hardware constraint unless you have benchmark evidence.
SPLIT_TIME="${SPLIT_TIME:-06:00:00}"

# Discover all unique dataset names
DATASETS=($(ls "$DATA_DIR" | grep -E '_training|_test' | sed -E 's/(_training|_test).*//' | sort | uniq))

mkdir -p logs

for dataset in "${DATASETS[@]}"; do
  for splitter in "${SPLITTERS[@]}"; do
    output="results/${dataset}/${splitter}/split.json"
    mkdir -p "$(dirname "$output")"
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=split_${dataset}_${splitter}
#SBATCH --output=logs/split_${dataset}_${splitter}.out
#SBATCH --error=logs/split_${dataset}_${splitter}.err
#SBATCH --time=${SPLIT_TIME}
#SBATCH --mem=20G
#SBATCH --cpus-per-task=1

module load julia
julia --compiled-modules=existing scripts/run_split.jl $dataset $splitter $output
EOF
  done
done
