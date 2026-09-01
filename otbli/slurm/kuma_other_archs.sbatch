#!/bin/bash
# Second-architecture runs on Kuma: the theory-transfer control (GPT-2) and the
# SwiGLU extension (TinyLlama -> Mistral-7B). Prefetch on the frontend first:
#
#   MODELS="openai-community/gpt2-large TinyLlama/TinyLlama_v1.1 mistralai/Mistral-7B-v0.1" \
#       bash slurm/setup_kuma.sh
#   sbatch slurm/kuma_other_archs.sbatch
#
# Llama-2/3 work identically but are license-gated on the Hub: on the frontend run
# `huggingface-cli login` with an approved token, add e.g. meta-llama/Llama-2-7b-hf
# to MODELS above and to the run line below.
#
#SBATCH --job-name=otbli-arch2
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16        # Kuma hard cap: max 16 cores per GPU
#SBATCH --mem=88G                 # Kuma cap at 16 cores: 16*5900MB & 92GB max
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
SCRATCH_DIR=${SCRATCH_DIR:-/scratch/$USER}
export HF_HOME=$SCRATCH_DIR/hf_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

module purge 2>/dev/null || true
module load gcc python 2>/dev/null || true
source "$SCRATCH_DIR/otbli-venv/bin/activate"

python -c "import torch; print('torch', torch.__version__, '| gpu:', torch.cuda.get_device_name(0))"

python scripts/run_all.py --models openai-community/gpt2-large \
    TinyLlama/TinyLlama_v1.1 mistralai/Mistral-7B-v0.1 \
    --device cuda --out results
