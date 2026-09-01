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
#
# If prefetching fails with "429 Too Many Requests" / "rate limit your IP":
# the cluster's shared egress IP has hit HF's anonymous rate limit. Create a
# free token at https://huggingface.co/settings/tokens (Read access is
# enough), then either `huggingface-cli login` or:
#   export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
# and re-run this script -- already-fetched files are cached and skipped.
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
echo "   (loaded through the same loader the jobs use, so only the files"
echo "    actually needed are fetched -- not onnx/tf/flax siblings; retries"
echo "    with backoff on transient rate limits / 429s)"
export REPO_DIR
python - << 'PY'
import gc
import os
import sys
import time

sys.path.insert(0, os.environ["REPO_DIR"])
from otbli import load_model


def fetch(name, revision=None, attempts=5):
    label = name + (f" @ {revision}" if revision else "")
    for i in range(attempts):
        try:
            print(f"   prefetch {label} (try {i + 1}/{attempts})", flush=True)
            model, tok = load_model(name, device="cpu", revision=revision)
            del model, tok
            gc.collect()
            return
        except Exception as e:
            wait = min(30 * (2 ** i), 300)
            print(f"     failed: {e}\n     retrying in {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"could not fetch {label} after {attempts} attempts -- "
                       "if this is a 429, set HF_TOKEN (see setup_kuma.sh header) "
                       "and re-run; already-fetched files are cached and skipped")


for m in os.environ["MODELS"].split():
    fetch(m)
tm = os.environ.get("TRAJ_MODEL", "").strip()
if tm:
    for s in os.environ["TRAJ_STEPS"].split():
        fetch(tm, revision=f"step{s}")
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
