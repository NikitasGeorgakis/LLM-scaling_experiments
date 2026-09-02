#!/bin/bash
# Scaling sweep -- the trend across sizes is the scientifically decisive output.
mkdir -p logs results
DATASET=${1:-pile}
for M in pythia-410m pythia-1b pythia-1.4b pythia-2.8b; do
    sbatch --job-name=${M} slurm_pythia.sh EleutherAI/${M} ${DATASET}
done
echo "Submitted. Monitor: squeue -u \$USER   Summarize: python summarize.py"
