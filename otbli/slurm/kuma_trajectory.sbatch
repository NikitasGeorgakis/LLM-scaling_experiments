#!/bin/bash
# Checkpoint-trajectory follow-up (Section 6.5) on Kuma — the slack-hypothesis
# test at the new scale. Prefetch the revisions first, on the frontend:
#
#     TRAJ_MODEL=EleutherAI/pythia-6.9b bash slurm/setup_kuma.sh
#     sbatch slurm/kuma_trajectory.sbatch
#
# Override the model at submit time with:
#     sbatch --export=ALL,MODEL=EleutherAI/pythia-2.8b slurm/kuma_trajectory.sbatch
#
#SBATCH --job-name=otbli-traj
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16        # Kuma hard cap: max 16 cores per GPU
#SBATCH --mem=88G                 # Kuma cap at 16 cores: 16*5900MB & 92GB max
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
MODEL=${MODEL:-EleutherAI/pythia-6.9b}
SCRATCH_DIR=${SCRATCH_DIR:-/scratch/$USER}
export HF_HOME=$SCRATCH_DIR/hf_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

module purge 2>/dev/null || true
module load gcc python 2>/dev/null || true
source "$SCRATCH_DIR/otbli-venv/bin/activate"

python -c "import torch; print('torch', torch.__version__, '| gpu:', torch.cuda.get_device_name(0))"

python scripts/run_trajectory.py --model "$MODEL" \
    --steps 512 1000 2000 4000 8000 16000 32000 64000 143000 \
    --device cuda --dataset-label pile --out results
