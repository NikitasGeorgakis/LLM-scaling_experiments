#!/bin/bash
#SBATCH --job-name=otd-e5
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

echo "=== poolA (missing prerequisite from Step 6) ==="
python3 run_screen.py --model-key gpt2-large --construction barycenter \
    --pool "$POOL_A" --refine --device cuda --out runs/e5_poolA

echo "=== poolD ==="
python3 run_screen.py --model-key gpt2-large --construction barycenter \
    --pool pools/D_sel.jsonl --device cuda --out runs/e5_poolD

echo "=== poolE ==="
python3 run_screen.py --model-key gpt2-large --construction barycenter \
    --pool pools/E_sel.jsonl --device cuda --out runs/e5_poolE

echo "=== max-t combine ==="
python3 run_maxt.py \
    --records runs/e5_poolA/screen.npz runs/e5_poolD/screen.npz runs/e5_poolE/screen.npz \
    --out runs/e5_maxt

echo "=== E5 DONE ==="
