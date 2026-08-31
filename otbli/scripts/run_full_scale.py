#!/usr/bin/env python3
"""Full-scale evaluation on pretrained Pythia checkpoints (paper Section 6).

Per model: build pools -> base losses -> Stage-A position screen (gamma = 1)
-> Stage-B gate machinery at the top-2 positions -> mechanism diagnostics
(exact recovery, drift identity, matching diagnostic, baseline ordering)
-> pre-declared loss-only secondary candidates (reported, NOT auto-confirmed:
the single confirmation is a deliberate, separate act via run_confirmation.py).

Example (reproduces Tables 5-6 and the diagnostics of Section 6.4):
    python scripts/run_full_scale.py \
        --models EleutherAI/pythia-410m EleutherAI/pythia-1b \
                 EleutherAI/pythia-1.4b EleutherAI/pythia-2.8b \
        --device cuda --out results/
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from otbli import load_model
from otbli.arch import get_layers
from otbli.config import (OTConfig, GateConfig, DataConfig, ProtocolConfig,
                          print_registered)
from otbli.data import load_or_build_pools
from otbli.metrics import batch_losses
from otbli.protocol import (stage_a_screen, stage_b_gate, baseline_deltas,
                            loss_only_candidate)
from otbli.diagnostics import (matching_diagnostic, state_fingerprint,
                               fingerprints_equal, exact_recovery_check,
                               drift_identity_check)
from otbli.protocol import make_insertion


def run_one_model(name, args, ot_cfg, gate_cfg, data_cfg, proto_cfg):
    short = name.split("/")[-1]
    print(f"\n{'#' * 78}\n# {name}\n{'#' * 78}", flush=True)
    t_start = time.time()

    model, tok = load_model(name, device=args.device)
    layers = get_layers(model)
    L = len(layers)
    n_par = sum(p.numel() for p in model.parameters())
    from otbli.atomize import _detect_mlp_type
    arch = _detect_mlp_type(layers[0])
    print(f"blocks L = {L}, parameters = {n_par / 1e6:.0f}M, MLP family = {arch}" + (" (gauge-fixed SwiGLU atoms)" if arch == "gated" else ""))

    pools_path = os.path.join(args.out, f"pools_{short}_seed{data_cfg.seed}.pt")
    sel, conf = load_or_build_pools(tok, data_cfg, pools_path)
    print(f"pools: selection {tuple(sel.shape)}  confirmation {tuple(conf.shape)} "
          f"(cached at {pools_path}; confirmation stays untouched)")

    # reference state for the exact-recovery diagnostic (Section 6.4 (i))
    fp0 = state_fingerprint(model)
    ref_logits = exact_recovery_check(model, sel[0], args.device)

    base_sel = batch_losses(model, sel, args.device)
    print(f"base selection loss L(0) = {base_sel.mean():.4f} nats/token "
          f"(PPL = {np.exp(base_sel.mean()):.2f})")

    positions = args.positions if args.positions else None
    screen = stage_a_screen(model, sel, base_sel, args.device, ot_cfg,
                            proto_cfg, positions=positions)
    top = screen[:proto_cfg.top_k_positions]
    print(f"[Stage A] top-{len(top)} positions: "
          + ", ".join(f"(F{r['pair'][0]},F{r['pair'][1]})" for r in top))

    stage_b = [stage_b_gate(model, sel, base_sel, args.device, r["i"], ot_cfg,
                            gate_cfg, proto_cfg) for r in top]

    # ---------------------------------------------------------- diagnostics
    i_top = top[0]["i"]
    diag = {}
    diag["matching"] = matching_diagnostic(layers[i_top], layers[i_top + 1],
                                           max_units=proto_cfg.match_max_units)
    print(f"[diag] matching at (F{i_top+1},F{i_top+2}): "
          f"{diag['matching']['frac_rematched']:.1%} of "
          f"{diag['matching']['units']} units re-matched; mean pairing cost "
          f"{diag['matching']['mean_cost_identity']:.3f} -> "
          f"{diag['matching']['mean_cost_optimal']:.3f} "
          f"({diag['matching']['reduction']:.1%} reduction)")
    if "no_gauge" in diag["matching"]:
        ng = diag["matching"]["no_gauge"]
        print(f"[diag] gauge ablation (Sec. 3.6 rem. (i)): matched-cost "
              f"reduction {diag['matching']['reduction']:.1%} gauge-fixed vs "
              f"{ng['reduction']:.1%} raw; rematched "
              f"{diag['matching']['frac_rematched']:.1%} vs {ng['frac_rematched']:.1%}")

    ins = make_insertion(model, i_top, ot_cfg, args.device)
    d1 = stage_b[0]["drift1"]
    diag["drift_identity"] = drift_identity_check(model, ins, sel, args.device, d1)
    print(f"[diag] drift identity at gamma=0.1: measured "
          f"{diag['drift_identity']['measured']:.6e} vs predicted "
          f"{diag['drift_identity']['predicted']:.6e} "
          f"(rel err {diag['drift_identity']['rel_err']:.2e})")
    ins.remove()
    del ins

    diag["baselines"] = baseline_deltas(model, sel, base_sel, args.device,
                                        i_top, ot_cfg.tau, proto_cfg.n_boot_sel)
    bl = diag["baselines"]
    bary1 = stage_b[0]["records"].get(1.0, {}).get("dL_raw", float("nan"))
    print(f"[diag] gamma=1 ordering at (F{i_top+1},F{i_top+2}): barycentric "
          f"{bary1:+.4f} <= naive {bl['naive_average']['dL_gamma1']:+.4f} "
          f"< duplicate {bl['duplicate']['dL_gamma1']:+.4f}")

    diag["exact_recovery_logits"] = exact_recovery_check(model, sel[0],
                                                         args.device, ref_logits)
    diag["exact_recovery_weights"] = fingerprints_equal(fp0, state_fingerprint(model))
    print(f"[diag] exact recovery: logits bit-equal = "
          f"{diag['exact_recovery_logits']}, weights untouched = "
          f"{diag['exact_recovery_weights']}")

    cand = loss_only_candidate(stage_b, gate_cfg, proto_cfg.n_boot_sel)

    # ------------------------------------------------------------- verdict
    best = min(stage_b, key=lambda r: r["records"][r["gamma_hat"]]["J"])
    gamma_star = best["gamma_star"]
    print(f"\n>>> VERDICT [{short}]  gamma* = {gamma_star}"
          + ("  ->  M+_gamma* = M exactly (safe fallback fired)"
             if gamma_star == 0.0 else
             f"  ->  retained; dL = {best['records'][gamma_star]['dL_raw']:+.3e} nats"))
    if cand:
        print(f"    loss-only secondary candidate: pair (F{cand['pair'][0]},"
              f"F{cand['pair'][1]}), gamma = {cand['gamma']}, "
              f"dL = {cand['dL']:+.3e}, CI95 = [{cand['ci'][0]:+.2e}, "
              f"{cand['ci'][1]:+.2e}], strength = {cand['strength']:.2f}")
    else:
        print("    loss-only secondary candidate: none "
              "(dL > 0 for every stability-feasible gamma > 0)")

    out = {"model": name, "blocks": L, "params": n_par,
           "L0_sel": float(base_sel.mean()),
           "base_sel_losses": base_sel.tolist(),
           "stage_a": screen, "stage_b": stage_b,
           "diagnostics": diag,
           "loss_only_candidate": cand,
           "gamma_star": gamma_star,
           "wallclock_s": time.time() - t_start}
    path = os.path.join(args.out, f"full_scale_{short}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"    results written to {path}  ({out['wallclock_s']:.0f}s)")

    del model
    if str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=[
        "EleutherAI/pythia-410m", "EleutherAI/pythia-1b",
        "EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dataset", default=None,
                    help="HF dataset name or local .jsonl of held-out Pile text")
    ap.add_argument("--positions", nargs="*", type=int, default=None,
                    help="restrict the Stage-A screen to these 0-indexed positions")
    ap.add_argument("--kl-batches", type=int, default=None)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ot_cfg, gate_cfg = OTConfig(), GateConfig()
    data_cfg, proto_cfg = DataConfig(), ProtocolConfig()
    if args.dataset:
        data_cfg.dataset = args.dataset
    if args.kl_batches:
        proto_cfg.kl_batches = args.kl_batches
    print_registered(ot_cfg, gate_cfg, data_cfg, proto_cfg)

    torch.manual_seed(data_cfg.seed)
    summary = []
    for name in args.models:
        res = run_one_model(name, args, ot_cfg, gate_cfg, data_cfg, proto_cfg)
        summary.append({"model": name, "gamma_star": res["gamma_star"],
                        "L0_sel": res["L0_sel"],
                        "screened_pair": res["stage_b"][0]["pair"]})
    print("\n" + "=" * 78 + "\nSUMMARY (cf. Table 5)")
    for s in summary:
        print(f"  {s['model']:<28s} L(0)={s['L0_sel']:.4f}  "
              f"screened (F{s['screened_pair'][0]},F{s['screened_pair'][1]})  "
              f"gamma* = {s['gamma_star']}")


if __name__ == "__main__":
    main()
