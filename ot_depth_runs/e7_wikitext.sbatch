#!/bin/bash
#SBATCH --job-name=otd-e7wiki
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --time=03:00:00
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

python3 run_screen.py --model-key gpt2-large --construction barycenter \
    --pool pools/G_wikitext.jsonl --refine --device cuda --out runs/e7_wikitext

echo "=== E7 WIKITEXT DONE ==="
