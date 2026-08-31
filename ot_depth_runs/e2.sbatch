#!/bin/bash
#SBATCH --job-name=otd-e2
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=88G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source /scratch/georgaki/otbli-venv/bin/activate
export HF_HOME=/scratch/georgaki/hf_cache
export POOL_A=$PWD/pools/pool_A.jsonl
export POOL_B=$PWD/pools/pool_B.jsonl

for M in pythia-410m pythia-1b pythia-1.4b pythia-2.8b; do
    echo "=== $M ==="
    python3 run_screen.py --model-key "$M" --construction barycenter \
        --pool "$POOL_A" --refine --confirm-pool "$POOL_B" \
        --device cuda --out "runs/e2_${M}"
done

echo "=== E2 DONE ==="
