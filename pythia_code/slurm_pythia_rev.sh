#!/bin/bash
#SBATCH --partition=h100
#SBATCH --job-name=p14b-loc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=20GB
#SBATCH --time=01:00:00
#SBATCH --output=logs/p14b-loc-%j.out
#SBATCH --error=logs/p14b-loc-%j.err

set -e

source /scratch/georgaki/pythia_venv/bin/activate
export HF_HOME=/scratch/georgaki/hf_cache

cd ~/pythia_code
mkdir -p logs results

REV=${1:-step1000}
OUTDIR="/home/georgaki/pythia_code/results"

python pythia_select.py \
    --model EleutherAI/pythia-1.4b \
    --revision "${REV}" \
    --cache-dir /scratch/georgaki/hf_cache \
    --out "${OUTDIR}/p14b-${REV}.json"

echo "✓ Completed for revision: ${REV}"
