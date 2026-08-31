#!/usr/bin/env python3
"""The single, pre-declared out-of-sample confirmation (Sections 3.4, 6.3).

WARNING: by pre-registration the confirmation pool may be touched EXACTLY ONCE
per candidate family. Decide the candidate (i*, gamma*) from the selection-set
records BEFORE running this, and report the outcome whatever it is.

Rule (3.48): accept iff  mean(d) < -delta  AND  CI_upper_95%(d) < 0,
with d the per-batch paired loss differences on the confirmation pool,
delta = 1e-3 nats, and a 10,000-resample nonparametric bootstrap.

Example (the Section 6.3 candidate):
    python scripts/run_confirmation.py --model EleutherAI/pythia-1.4b \
        --position 1 --gamma 0.01 --pools results/pools_pythia-1.4b_seed1234.pt
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from otbli import load_pythia
from otbli.config import OTConfig, GateConfig, DataConfig, print_registered
from otbli.data import load_or_build_pools
from otbli.protocol import run_confirmation


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--position", type=int, required=True,
                    help="0-indexed i of the pair (layers[i], layers[i+1]); "
                         "the paper's 1-indexed pair (F_{i+1}, F_{i+2})")
    ap.add_argument("--gamma", type=float, required=True)
    ap.add_argument("--pools", required=True,
                    help=".pt pools cache written by run_full_scale.py — the "
                         "confirmation pool must be the same untouched one")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ot_cfg, gate_cfg, data_cfg = OTConfig(), GateConfig(), DataConfig()
    print_registered(ot_cfg, gate_cfg)

    model, tok = load_pythia(args.model, device=args.device, revision=args.revision)
    _, conf = load_or_build_pools(tok, data_cfg, args.pools)
    print(f"confirmation pool {tuple(conf.shape)} — evaluated exactly once.")

    res = run_confirmation(model, conf, args.device, args.position, args.gamma,
                           ot_cfg, gate_cfg, n_boot=10000)
    print(f"\npair (F{res['pair'][0]},F{res['pair'][1]}), gamma = {res['gamma']}")
    print(f"  mean d      = {res['d_mean']:+.3e} nats")
    print(f"  CI95        = [{res['ci'][0]:+.3e}, {res['ci'][1]:+.3e}]")
    print(f"  t = {res['t']:+.2f},  p_one-sided = {res['p_one_sided']:.3f}")
    print(f"  rule (3.48) verdict: "
          f"{'ACCEPTED (material, significant out of sample)' if res['accepted'] else 'NOT CONFIRMED'}")

    short = args.model.split("/")[-1]
    tag = f"{short}_i{args.position}_g{args.gamma}"
    path = os.path.join(args.out, f"confirmation_{tag}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"  written to {path}")


if __name__ == "__main__":
    main()
