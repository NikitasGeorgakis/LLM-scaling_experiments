#!/bin/bash
#SBATCH --partition=h100
#SBATCH --job-name=eval-pythia
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval-%j.out

python pythia_select.py --model EleutherAI/pythia-1.4b --out results/pythia_1.4b.json
