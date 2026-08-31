#!/usr/bin/env python3
"""Corrected out-of-sample confirmation for OT barycentric layer insertion.

This script evaluates ONE pre-selected candidate (position, gamma) on the
untouched confirmation pool. It performs no tuning and no model-parameter
updates.

Confirmation criterion:
  mean paired loss difference < 0
  AND upper endpoint of the 95% bootstrap CI < 0
  AND output KL <= eps_KL
  AND representation drift <= eps_rep

No materiality delta is used.
"""

import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from otbli import load_model
from otbli.config import OTConfig, DataConfig
from otbli.data import load_or_build_pools
from otbli.metrics import batch_losses, batch_kl, bootstrap_ci, paired_t
from otbli.protocol import make_insertion, measure_drift1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--position",
        type=int,
        required=True,
        help="0-indexed i for pair (layers[i], layers[i+1]); "
             "paper notation is (F_{i+1}, F_{i+2})",
    )
    ap.add_argument("--gamma", type=float, required=True)
    ap.add_argument("--pools", required=True)
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--eps-kl", type=float, default=0.05)
    ap.add_argument("--eps-rep", type=float, default=0.05)
    ap.add_argument("--drift-batches", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    ot_cfg = OTConfig()
    data_cfg = DataConfig()

    print("=" * 80)
    print("CORRECTED OUT-OF-SAMPLE CONFIRMATION")
    print("=" * 80)
    print(f"model        : {args.model}")
    print(
        f"position     : {args.position} "
        f"(pair F{args.position + 1},F{args.position + 2})"
    )
    print(f"gamma        : {args.gamma}")
    print(f"bootstrap    : {args.n_boot}")
    print(f"eps_KL       : {args.eps_kl}")
    print(f"eps_rep      : {args.eps_rep}")
    print("IMPORTANT    : position and gamma are FIXED; no tuning is performed")
    print("=" * 80, flush=True)

    model, tok = load_model(args.model, device=args.device)
    _sel, conf = load_or_build_pools(tok, data_cfg, args.pools)

    print(f"confirmation pool shape: {tuple(conf.shape)}", flush=True)

    ins = make_insertion(
        model,
        args.position,
        ot_cfg,
        args.device,
        verbose=False,
    )

    # Representation-drift stability on confirmation data.
    drift1 = measure_drift1(
        model,
        ins,
        conf,
        args.device,
        min(args.drift_batches, len(conf)),
    )
    rep = float(args.gamma * args.gamma * drift1)

    # Base loss on untouched confirmation data.
    ins.set_gamma(0.0)
    base_losses = batch_losses(model, conf, args.device)

    # Fixed candidate on the same paired batches.
    ins.set_gamma(args.gamma)
    cand_losses = batch_losses(model, conf, args.device)

    d = cand_losses - base_losses
    d_mean = float(d.mean())
    ci_lo, ci_hi = bootstrap_ci(d, args.n_boot, seed=99)
    t_stat, p_one = paired_t(d)

    base_loss = float(base_losses.mean())
    cand_loss = float(cand_losses.mean())

    # Output-distribution stability on confirmation data.
    ins.set_gamma(args.gamma)
    kl = float(batch_kl(model, ins, conf, args.device))

    ins.remove()

    ppl_base = math.exp(base_loss)
    ppl_cand = math.exp(cand_loss)

    rel_nll_improvement = (
        (base_loss - cand_loss) / abs(base_loss)
        if base_loss != 0 else float("nan")
    )
    rel_ppl_improvement = (
        (ppl_base - ppl_cand) / ppl_base
        if ppl_base != 0 else float("nan")
    )

    confirmed = bool(
        d_mean < 0.0
        and ci_hi < 0.0
        and kl <= args.eps_kl
        and rep <= args.eps_rep
    )

    result = {
        "model": args.model,
        "position": args.position,
        "pair": [args.position + 1, args.position + 2],
        "gamma": args.gamma,
        "n_confirmation_batches": int(len(conf)),
        "base_loss": base_loss,
        "candidate_loss": cand_loss,
        "d_mean": d_mean,
        "ci95": [float(ci_lo), float(ci_hi)],
        "t_stat": float(t_stat),
        "p_one_sided": float(p_one),
        "output_KL": kl,
        "representation_drift": rep,
        "D_rep_1": float(drift1),
        "eps_KL": args.eps_kl,
        "eps_rep": args.eps_rep,
        "relative_nll_improvement": float(rel_nll_improvement),
        "base_ppl": float(ppl_base),
        "candidate_ppl": float(ppl_cand),
        "relative_ppl_improvement": float(rel_ppl_improvement),
        "confirmed": confirmed,
        "paired_differences": d.tolist(),
    }

    print()
    print("=" * 80)
    print("CONFIRMATION RESULT")
    print("=" * 80)
    print(f"base loss       = {base_loss:.8f}")
    print(f"candidate loss  = {cand_loss:.8f}")
    print(f"paired dL       = {d_mean:+.8e}")
    print(f"bootstrap CI95  = [{ci_lo:+.8e}, {ci_hi:+.8e}]")
    print(f"t statistic     = {t_stat:+.4f}")
    print(f"one-sided p     = {p_one:.6g}")
    print(f"output KL       = {kl:.8e}")
    print(f"representation  = {rep:.8e}")
    print(f"base PPL        = {ppl_base:.6f}")
    print(f"candidate PPL   = {ppl_cand:.6f}")
    print(f"NLL improvement = {100 * rel_nll_improvement:.4f}%")
    print(f"PPL improvement = {100 * rel_ppl_improvement:.4f}%")
    print()
    print(
        "VERDICT:",
        "CONFIRMED OUT OF SAMPLE" if confirmed else "NOT CONFIRMED",
    )
    print("=" * 80, flush=True)

    short = args.model.split("/")[-1]
    gamma_tag = str(args.gamma).replace(".", "p")
    tag = f"{short}_i{args.position}_g{gamma_tag}"

    json_path = os.path.join(
        args.out,
        f"confirmation_{tag}.json",
    )
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"JSON -> {json_path}", flush=True)


if __name__ == "__main__":
    main()
