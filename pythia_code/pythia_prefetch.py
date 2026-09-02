"""
One-time pre-fetch of every Pythia checkpoint (+ tokenizer) and both
validation datasets into the shared cache at $HF_HOME (set by slurm_pythia.sh
to /scratch/$USER/hf_cache).

WHY THIS SCRIPT EXISTS: submit_all.sh launches four SLURM jobs (410m, 1b,
1.4b, 2.8b) that can start within seconds of each other. All four read the
SAME two datasets. Without a pre-fetch step they would race to download the
same files into the same cache directory on first use -- wasteful, and it
burns PAID GPU-hour on plain network I/O instead of on the actual experiment.
Run this once, on the login node if possible (free), before submit_all.sh.

AUTHENTICATION: none needed. EleutherAI/pythia-*, NeelNanda/pile-10k, and
wikitext are all public, ungated repositories -- no HF account, no token,
no `huggingface-cli login`.

This also doubles as a pre-flight check: it calls the exact same
`load_dataset(...)` used later by pythia_data.py, so if something about
dataset loading is going to fail, it fails here for free instead of inside a
billed GPU job.

Usage (try the login node first -- it is free, this is pure network I/O,
no GPU is touched):
    python pythia_prefetch.py

If the login node has no outbound internet (some clusters restrict this;
Kuma's probe confirmed COMPUTE nodes can reach huggingface.co, the login
node was not separately tested), you will see a connection/timeout error.
In that case, fall back to a short interactive GPU session:
    srun --partition=h100 --gpus=1 --time=00:30:00 --pty bash
    source /scratch/$USER/pythia_venv/bin/activate
    export HF_HOME=/scratch/$USER/hf_cache
    python pythia_prefetch.py
    exit
"""
import argparse
import os
import subprocess
import time

os.environ.setdefault("HF_HOME", f"/scratch/{os.environ.get('USER', 'user')}/hf_cache")

MODELS = ["EleutherAI/pythia-70m", "EleutherAI/pythia-410m",
          "EleutherAI/pythia-1b", "EleutherAI/pythia-1.4b",
          "EleutherAI/pythia-2.8b"]
# (repo_id, config_name_or_None, split)
DATASETS = [("NeelNanda/pile-10k", None, "train"),
            ("wikitext", "wikitext-103-raw-v1", "validation")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-models", action="store_true")
    ap.add_argument("--skip-datasets", action="store_true")
    args = ap.parse_args()

    print(f"HF_HOME = {os.environ['HF_HOME']}\n")

    if not args.skip_models:
        from huggingface_hub import snapshot_download
        for name in MODELS:
            t0 = time.time()
            print(f"[model] {name} ...", flush=True)
            path = snapshot_download(repo_id=name)
            print(f"        cached at {path}  ({time.time()-t0:.0f}s)")

    if not args.skip_datasets:
        from datasets import load_dataset
        for name, config, split in DATASETS:
            t0 = time.time()
            print(f"[dataset] {name} ({config or 'default'}) split={split} ...",
                  flush=True)
            # trust_remote_code=True is defensive: some canonical HF datasets
            # (wikitext historically among them) ship a loading script rather
            # than plain data files, and recent `datasets` versions require
            # explicit trust for that. Harmless no-op if not needed.
            try:
                ds = (load_dataset(name, config, split=split, trust_remote_code=True)
                     if config else
                     load_dataset(name, split=split, trust_remote_code=True))
            except TypeError:
                # older `datasets` versions do not accept trust_remote_code
                ds = (load_dataset(name, config, split=split) if config
                     else load_dataset(name, split=split))
            print(f"          {len(ds):,} rows  ({time.time()-t0:.0f}s)")

    out = subprocess.run(["du", "-sh", os.environ["HF_HOME"]],
                         capture_output=True, text=True)
    print(f"\nTotal cache size: {out.stdout.strip()}")
    print("Cached. select/confirm and --smoke runs will not re-download anything.")


if __name__ == "__main__":
    main()
