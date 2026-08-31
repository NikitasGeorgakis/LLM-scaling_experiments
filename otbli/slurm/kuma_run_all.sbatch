#!/bin/bash
# Full unattended sweep on SCITAS Kuma (submit FROM THE REPO ROOT after
# running slurm/setup_kuma.sh once on the frontend):
#
#     sbatch slurm/kuma_run_all.sbatch
#
# The job runs on the cluster and is completely independent of your own
# computer — you can log out and shut your machine down. Monitor with
# `squeue -u $USER` and `tail -f logs/otbli-all_<jobid>.out`.
#
#SBATCH --job-name=otbli-all
#SBATCH --partition=h100          # Kuma has no default partition: h100 or l40s
#SBATCH --qos=normal              # up to 3 days; use 'long' for up to 7
#SBATCH --time=12:00:00           # default walltime is 5 min — always set this
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G                # host RAM for fp32 checkpoint loading
#SBATCH --gres=gpu:1              # one H100 94GB — 6.9B fp32 fits comfortably
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
SCRATCH_DIR=${SCRATCH_DIR:-/scratch/$USER}
export HF_HOME=$SCRATCH_DIR/hf_cache
export HF_HUB_OFFLINE=1           # everything was prefetched by setup_kuma.sh
export TOKENIZERS_PARALLELISM=false

module purge 2>/dev/null || true
module load gcc python 2>/dev/null || true
source "$SCRATCH_DIR/otbli-venv/bin/activate"

python -c "import torch; print('torch', torch.__version__, '| gpu:', torch.cuda.get_device_name(0))"

# Resume-safe: models with an existing results/full_scale_<model>.json are
# skipped, so if the job dies you just resubmit the same script.
python scripts/run_all.py --device cuda --out results

# Optional extras (uncomment):
#   pythia-12b (48 GB fp32 weights, fits the 94 GB H100; raise --time):
# python scripts/run_all.py --models EleutherAI/pythia-12b --device cuda --out results
