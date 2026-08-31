#!/bin/bash
#SBATCH --job-name=otd-e3
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --time=06:00:00
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

python3 run_screen.py --model-key gpt2-large --construction copy_next \
    --pool "$POOL_A" --refine --device cuda --out runs/e3_copy_next

python3 run_screen.py --model-key gpt2-large --construction hard_ot \
    --pool "$POOL_A" --refine --device cuda --out runs/e3_hard_ot

echo "=== E3 DONE ==="
