#!/bin/bash
#SBATCH --partition=l40s
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2880M
#SBATCH --time=00:30:00
#SBATCH --output=logs/download_%j.out

export HF_HOME=/scratch/$USER/hf_cache
source /scratch/$USER/pythia_venv/bin/activate

python << 'PEOF'
from transformers import AutoModel
for rev in ['step1000', 'step8000', 'step32000', 'step72000']:
    print(f'\n=== pythia-1.4b@{rev} ===')
    try:
        AutoModel.from_pretrained('EleutherAI/pythia-1.4b', revision=rev, 
                                  cache_dir='/scratch/$USER/hf_cache')
        print(f'OK')
    except Exception as e:
        print(f'FAIL: {str(e)[:200]}')
PEOF
