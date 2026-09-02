#!/bin/bash
# Run ONCE:   bash setup_env.sh
# Creates a venv on /scratch with everything needed. Try the login node first
# (free). If pip fails there with network/timeout errors, the login node has no
# outbound internet -- rerun inside a cheap interactive session instead:
#   srun --partition=l40s --gpus=1 --time=00:40:00 --pty bash setup_env.sh
set -e

VENV=/scratch/$USER/pythia_venv
export HF_HOME=/scratch/$USER/hf_cache

echo "==> python: $(python3 --version)"
echo "==> creating venv at $VENV"
mkdir -p /scratch/$USER
python3 -m venv $VENV
source $VENV/bin/activate
python -m pip install --upgrade pip wheel

# PyPI linux torch wheels bundle CUDA 12 -- no special index needed. On Kuma's
# Python 3.9, pip will automatically resolve the newest versions that still
# ship 3.9 wheels; the code is tested to be compatible across that range
# (the insertion hook is deliberately version-agnostic).
echo "==> installing torch + libraries"
pip install torch
pip install "transformers>=4.40,<6" datasets scipy numpy

echo "==> versions"
python - << 'PY'
import torch, transformers, datasets, scipy, sys
print("python      ", sys.version.split()[0])
print("torch       ", torch.__version__, "| cuda build:", torch.version.cuda)
print("transformers", transformers.__version__)
print("datasets    ", datasets.__version__)
print("scipy       ", scipy.__version__)
PY

mkdir -p $HF_HOME
echo ""
echo "Done. venv: $VENV   HF cache: $HF_HOME"
echo "NOTE: torch.cuda.is_available() is False on the login node (no GPU) -- expected."
