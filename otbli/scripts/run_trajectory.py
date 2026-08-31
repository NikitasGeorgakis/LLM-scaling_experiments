#!/usr/bin/env python3
"""Checkpoint-trajectory follow-up (the falsifiable prediction of Section 6.5).

The slack hypothesis predicts that the small-gamma loss dip at fixed
architecture decays monotonically with training progress. The Pythia suite
publishes intermediate checkpoints as HF revisions ('step512' ...
'step143000'); this script runs the UNCHANGED two-stage protocol at each one.

Outputs
  results/trajectory_<model>_<label>.csv           the compact schema
        step,gamma,delta_L,tau,model,dataset
    with gamma = gamma* of the pre-registered protocol and
    delta_L = paired selection-set dL(gamma*) in nats (0.0 whenever gamma*=0).
  results/trajectory_<model>_<label>_extended.csv  adds, per checkpoint, the
    loss-only diagnostic that Section 6.5 actually cares about: the most
    negative stability-feasible small-gamma dip and its bootstrap CI. Without
    it a run that nulls everywhere is indistinguishable from a flat dip curve.
  results/trajectory_<model>_step*.json            full per-checkpoint records.

Example (the run behind results/trajectory_pythia-1.4b_pile.csv):
    python scripts/run_trajectory.py --model EleutherAI/pythia-1.4b \
        --steps 512 1000 2000 4000 8000 16000 32000 64000 143000 \
        --device cuda --dataset-label pile
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from otbli import load_pythia
from otbli.config import (OTConfig, GateConfig, DataConfig, ProtocolConfig,
                          print_registered)
from otbli.data import load_or_build_pools
from otbli.metrics import batch_losses
from otbli.protocol import stage_a_screen, stage_b_gate, loss_only_candidate


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--steps", nargs="+", type=int,
                    default=[512, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 143000])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--dataset-label", default="pile",
                    help="label written to the CSV 'dataset' column")
    ap.add_argument("--positions", nargs="*", type=int, default=None,
                    help="optional restriction of the Stage-A screen")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ot_cfg, gate_cfg = OTConfig(), GateConfig()
    data_cfg, proto_cfg = DataConfig(), ProtocolConfig()
    if args.dataset:
        data_cfg.dataset = args.dataset
    print_registered(ot_cfg, gate_cfg, data_cfg, proto_cfg)

    short = args.model.split("/")[-1]
    csv_path = os.path.join(args.out, f"trajectory_{short}_{args.dataset_label}.csv")
    ext_path = os.path.join(args.out, f"trajectory_{short}_{args.dataset_label}_extended.csv")

    # pools are built once with the (revision-independent) tokenizer
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    pools_path = os.path.join(args.out, f"pools_{short}_seed{data_cfg.seed}.pt")
    sel, conf = load_or_build_pools(tok, data_cfg, pools_path)
    print(f"pools cached at {pools_path}; confirmation pool reserved untouched.")

    rows, ext_rows = [], []
    for step in args.steps:
        rev = f"step{step}"
        print(f"\n{'#' * 78}\n# {args.model} @ {rev}\n{'#' * 78}", flush=True)
        model, _ = load_pythia(args.model, device=args.device, revision=rev)
        base_sel = batch_losses(model, sel, args.device)
        print(f"base selection loss L(0) = {base_sel.mean():.4f} nats/token")

        screen = stage_a_screen(model, sel, base_sel, args.device, ot_cfg,
                                proto_cfg, positions=args.positions)
        top = screen[:proto_cfg.top_k_positions]
        stage_b = [stage_b_gate(model, sel, base_sel, args.device, r["i"],
                                ot_cfg, gate_cfg, proto_cfg) for r in top]

        best = min(stage_b, key=lambda r: r["records"][r["gamma_hat"]]["J"])
        g_star = best["gamma_star"]
        d_star = best["records"][g_star]["dL_raw"] if g_star in best["records"] else 0.0
        rows.append({"step": rev, "gamma": g_star, "delta_L": d_star,
                     "tau": ot_cfg.tau, "model": args.model,
                     "dataset": args.dataset_label})

        cand = loss_only_candidate(stage_b, gate_cfg, proto_cfg.n_boot_sel)
        ext_rows.append({
            "step": rev, "L0": float(base_sel.mean()),
            "gamma_star": g_star, "delta_L_star": d_star,
            "screened_pair": f"(F{best['pair'][0]},F{best['pair'][1]})",
            "lossonly_gamma": cand["gamma"] if cand else "",
            "lossonly_pair": (f"(F{cand['pair'][0]},F{cand['pair'][1]})" if cand else ""),
            "lossonly_dL": cand["dL"] if cand else "",
            "lossonly_ci_lo": cand["ci"][0] if cand else "",
            "lossonly_ci_hi": cand["ci"][1] if cand else "",
            "lossonly_sig_in_sample": cand["significant_in_sample"] if cand else "",
        })

        with open(os.path.join(args.out, f"trajectory_{short}_{rev}.json"), "w") as f:
            json.dump({"step": rev, "stage_a": screen, "stage_b": stage_b,
                       "loss_only_candidate": cand}, f, indent=2, default=str)

        # append-as-you-go so partial runs are never lost
        for path, data, fields in ((csv_path, rows, list(rows[0].keys())),
                                   (ext_path, ext_rows, list(ext_rows[0].keys()))):
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(data)

        del model
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()

    print(f"\ntrajectory written to {csv_path} and {ext_path}")


if __name__ == "__main__":
    main()
