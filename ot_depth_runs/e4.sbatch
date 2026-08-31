#!/bin/bash
#SBATCH --job-name=otd-e4
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --time=14:00:00
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

for S in step512 step1000 step2000 step4000 step8000 step16000 step32000 step64000 step143000; do
    echo "=== $S ==="
    python3 run_screen.py --model-key pythia-1.4b --revision "$S" \
        --construction barycenter --pool "$POOL_A" --refine \
        --confirm-pool "$POOL_B" --device cuda --out "runs/e4_${S}"
done

echo "=== E4 DONE ==="
