#!/bin/bash
# Corrected Step-1 all-position small-gamma screen on SCITAS Kuma.
#
# Submit from the repository root, after prefetching models + pools:
#
#   N=$(grep -vE '^[[:space:]]*(#|$)' configs/small_gamma_models.txt | wc -l)
#   sbatch --array=0-$((N-1))%2 slurm/kuma_small_gamma_array.sbatch
#
# Override publication/pilot settings at submission if desired:
#   sbatch --array=0-$((N-1))%2 \
#     --export=ALL,SCREEN_BATCHES=25,KL_BATCHES=25,DRIFT_BATCHES=4 \
#     slurm/kuma_small_gamma_array.sbatch
#
#SBATCH --job-name=otbli-sgamma
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=88G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

SCRATCH_DIR=${SCRATCH_DIR:-/scratch/$USER}
VENV=${VENV:-$SCRATCH_DIR/otbli-venv}
export HF_HOME=${HF_HOME:-$SCRATCH_DIR/hf_cache}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

MODELS_FILE=${MODELS_FILE:-configs/small_gamma_models.txt}
SCREEN_BATCHES=${SCREEN_BATCHES:-25}
KL_BATCHES=${KL_BATCHES:-25}
DRIFT_BATCHES=${DRIFT_BATCHES:-4}
RESULTS_DIR=${RESULTS_DIR:-results/small_gamma}
POOL_DIR=${POOL_DIR:-results}

module purge 2>/dev/null || true
module load gcc python 2>/dev/null || true
source "$VENV/bin/activate"
mkdir -p logs "$RESULTS_DIR"

mapfile -t MODELS < <(grep -vE '^[[:space:]]*(#|$)' "$MODELS_FILE")
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#MODELS[@]} )); then
    echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID but models file has ${#MODELS[@]} entries" >&2
    exit 2
fi
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"

echo "============================================================"
echo "job id          : $SLURM_JOB_ID"
echo "array task      : $SLURM_ARRAY_TASK_ID"
echo "host            : $(hostname)"
echo "model           : $MODEL"
echo "screen batches  : $SCREEN_BATCHES"
echo "KL batches      : $KL_BATCHES"
echo "drift batches   : $DRIFT_BATCHES"
echo "HF_HOME         : $HF_HOME"
echo "============================================================"

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda build:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0))
print("VRAM GiB:", torch.cuda.get_device_properties(0).total_memory / 2**30)
PY

srun python scripts/run_small_gamma_screen.py \
    --model "$MODEL" \
    --device cuda \
    --out "$RESULTS_DIR" \
    --pool-dir "$POOL_DIR" \
    --screen-batches "$SCREEN_BATCHES" \
    --kl-batches "$KL_BATCHES" \
    --drift-batches "$DRIFT_BATCHES"

echo "DONE: $MODEL"
