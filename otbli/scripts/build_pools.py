#!/usr/bin/env python3
"""Build and cache the selection/confirmation pools for each model ON A
MACHINE WITH INTERNET (e.g. the cluster login node): pool construction
streams held-out Pile text once, after which every compute job can run with
HF_HUB_OFFLINE=1. All Pythia sizes share the same tokenizer, so this is cheap.

    python scripts/build_pools.py --out results
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from otbli.config import DataConfig
from otbli.data import load_or_build_pools

DEFAULT_MODELS = [
    "EleutherAI/pythia-70m", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b", "EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoTokenizer
    cfg = DataConfig()
    if args.dataset:
        cfg.dataset = args.dataset
    for name in args.models:
        short = name.split("/")[-1]
        path = os.path.join(args.out, f"pools_{short}_seed{cfg.seed}.pt")
        tok = AutoTokenizer.from_pretrained(name)
        sel, conf = load_or_build_pools(tok, cfg, path)
        print(f"{name:<28s} pools sel {tuple(sel.shape)}  conf {tuple(conf.shape)}"
              f"  -> {path}")


if __name__ == "__main__":
    main()
