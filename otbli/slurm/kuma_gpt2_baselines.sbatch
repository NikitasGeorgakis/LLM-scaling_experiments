#!/bin/bash
#SBATCH --job-name=gpt2-baselines
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=88G
#SBATCH --gres=gpu:1
#SBATCH --chdir=/home/georgaki/otbli
#SBATCH --output=/home/georgaki/otbli/logs/gpt2_baselines_%j.out

set -euo pipefail

source /scratch/$USER/otbli-venv/bin/activate

export HF_HOME=/scratch/$USER/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p /home/georgaki/otbli/logs
mkdir -p /home/georgaki/otbli/results/gpt2_baseline_comparison

python scripts/run_gpt2_baseline_comparison.py \
  --model openai-community/gpt2-large \
  --position 11 \
  --pools results/pools_gpt2-large_seed1234.pt \
  --device cuda \
  --methods \
    barycenter \
    hard_ot_midpoint \
    naive_average \
    copy_prev \
    copy_next \
    random_native \
  --random-seeds 0 1 2 3 4 \
  --gamma-grid \
    0 \
    0.0001 \
    0.0003 \
    0.001 \
    0.003 \
    0.01 \
    0.03 \
    0.05 \
    0.075 \
    0.10 \
    0.125 \
    0.15 \
    0.20 \
    0.30 \
    0.50 \
  --n-boot-sel 5000 \
  --n-boot-conf 10000 \
  --kl-batches 25 \
  --kl-batches-conf 40 \
  --drift-batches 4 \
  --eps-kl 0.05 \
  --eps-rep 0.05 \
  --tau 0.5 \
  --eta 0.05 \
  --alt-rounds 25 \
  --sinkhorn-iters 80 \
  --ot-dtype float32 \
  --out results/gpt2_baseline_comparison
