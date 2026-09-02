#!/bin/bash
#SBATCH --job-name=otd-prereg410m
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

for S in step512 step1000 step2000 step4000 step8000 step16000 step32000 step64000 step143000; do
    echo "=== $S ==="
    python3 run_screen.py --model-key pythia-410m --revision "$S" \
        --construction barycenter --pool "$POOL_A" --refine \
        --device cuda --out "runs/prereg410m_${S}"
done

echo "=== PREREG 410M SCREEN DONE ==="
