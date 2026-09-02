#!/bin/bash
#SBATCH --job-name=pythia_ot
#SBATCH --partition=h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=5760M
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
set -euo pipefail
MODEL=${1:-EleutherAI/pythia-1b}
DATASET=${2:-pile}
EXTRA=${3:-}
TAG=$(basename $MODEL)

export HF_HOME=/scratch/$USER/hf_cache
export TOKENIZERS_PARALLELISM=false
source /scratch/$USER/pythia_venv/bin/activate

echo "== model: $MODEL  dataset: $DATASET  node: $SLURMD_NODENAME  job: $SLURM_JOB_ID =="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__, '| cuda', torch.cuda.is_available())"
mkdir -p results

echo "########## STAGE 1: SELECTION ##########"
python pythia_select.py --model "$MODEL" --dataset "$DATASET" $EXTRA \
    --out results/sel_${TAG}_${DATASET}.json

echo "########## STAGE 2: CONFIRMATION ##########"
python pythia_confirm.py --selection results/sel_${TAG}_${DATASET}.json \
    --exploratory 0.05,0.08 \
    --out results/conf_${TAG}_${DATASET}.json

echo "== finished: $(date) =="
