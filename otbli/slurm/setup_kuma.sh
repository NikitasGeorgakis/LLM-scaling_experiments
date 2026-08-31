#!/bin/bash
# One-time setup — run ON THE KUMA FRONTEND (kuma.hpc.epfl.ch), which has
# internet access. Creates a venv on /scratch, installs dependencies,
# prefetches all HuggingFace model weights (and, optionally, the trajectory
# checkpoints) into a scratch HF cache, and prebuilds the evaluation pools —
# so the sbatch jobs can then run fully offline (HF_HUB_OFFLINE=1).
#
#   bash slurm/setup_kuma.sh
#   TRAJ_MODEL=EleutherAI/pythia-6.9b bash slurm/setup_kuma.sh   # also prefetch
#                                          # the 9 trajectory revisions (~125 GB)
set -euo pipefail
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
SCRATCH_DIR=${SCRATCH_DIR:-/scratch/$USER}
VENV=${VENV:-$SCRATCH_DIR/otbli-venv}
export HF_HOME=${HF_HOME:-$SCRATCH_DIR/hf_cache}
export MODELS=${MODELS:-"EleutherAI/pythia-70m EleutherAI/pythia-160m EleutherAI/pythia-410m EleutherAI/pythia-1b EleutherAI/pythia-1.4b EleutherAI/pythia-2.8b EleutherAI/pythia-6.9b"}
export TRAJ_MODEL=${TRAJ_MODEL:-}
export TRAJ_STEPS=${TRAJ_STEPS:-"512 1000 2000 4000 8000 16000 32000 64000 143000"}

module purge 2>/dev/null || true
module load gcc python 2>/dev/null || echo "[warn] 'module load gcc python' failed — using system python3"

mkdir -p "$SCRATCH_DIR" "$HF_HOME" "$REPO_DIR/logs" "$REPO_DIR/results"
echo "== venv: $VENV"
python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip -q install --upgrade pip
pip -q install -r "$REPO_DIR/requirements.txt"
python - << 'PY'
import torch, transformers
print(f"== torch {torch.__version__} | transformers {transformers.__version__} | cuda build: {torch.version.cuda}")
PY

echo "== prefetching model weights into $HF_HOME"
python - << 'PY'
import os
from huggingface_hub import snapshot_download
for m in os.environ["MODELS"].split():
    print("   prefetch", m, flush=True)
    snapshot_download(m)
tm = os.environ.get("TRAJ_MODEL", "").strip()
if tm:
    for s in os.environ["TRAJ_STEPS"].split():
        print(f"   prefetch {tm} @ step{s}", flush=True)
        snapshot_download(tm, revision=f"step{s}")
PY

echo "== prebuilding selection/confirmation pools (streams held-out Pile once)"
python "$REPO_DIR/scripts/build_pools.py" --models $MODELS --out "$REPO_DIR/results"

echo
echo "Setup complete."
echo "  venv     : $VENV"
echo "  HF cache : $HF_HOME"
echo "Submit from the repo root:"
echo "  sbatch slurm/kuma_run_all.sbatch"
echo "  sbatch slurm/kuma_trajectory.sbatch     # optional (prefetch TRAJ_MODEL first)"
