#!/bin/bash
# Gated duplication judged on downstream task accuracy (scripts/run_task_gate.py).
# Prefetch the models and install lm-eval on the frontend first:
#
#   MODELS="EleutherAI/pythia-410m EleutherAI/pythia-1b" bash slurm/setup_kuma.sh
#   source /scratch/$USER/otbli-venv/bin/activate && pip install 'lm-eval>=0.4.2'
#   python scripts/run_task_gate.py --list-tasks bigbench | head -40   # verify names
#   sbatch slurm/kuma_task_gate.sbatch
#
# Task DATA is downloaded on first use, so this job does NOT set HF_HUB_OFFLINE.
# To run offline, do one --limit 8 pass on the frontend to populate the cache.
#
#SBATCH --job-name=otbli-task
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --time=12:00:00
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
export TOKENIZERS_PARALLELISM=false

module purge 2>/dev/null || true
module load gcc python 2>/dev/null || true
source "$SCRATCH_DIR/otbli-venv/bin/activate"

python -c "import torch; print('torch', torch.__version__, '| gpu:', torch.cuda.get_device_name(0))"

python scripts/run_task_gate.py \
    --models EleutherAI/pythia-410m EleutherAI/pythia-1b \
    --with-controls --device cuda --limit 200 --batch-size 16 --out results

# The confirmation is a SEPARATE, deliberate act -- decide (position, gamma)
# from the printed trade-off table, then run it exactly once:
# python scripts/run_task_gate.py --models EleutherAI/pythia-1b --confirm \
#     --position 0 --gamma 0.5 --device cuda --limit 200 --out results
