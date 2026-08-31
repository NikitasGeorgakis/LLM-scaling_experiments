#!/usr/bin/env python3
"""
Controlled GPT-2 baseline comparison for training-free gated layer insertion.

Purpose
-------
At a FIXED insertion position, compare alternative constructions of the newly
inserted layer under exactly the same gate-selection protocol.

Methods:
  barycenter       : free-support OT barycenter (current method)
  hard_ot_midpoint : one-to-one MLP assignment + matched midpoint
  naive_average    : direct parameter interpolation
  copy_prev        : verbatim previous layer
  copy_next        : verbatim next layer
  random_native_sK : model-native random initialization, K = random seed

For every method:
  1. Tune gamma ONLY on the selection pool.
  2. Require output-KL <= eps_KL and representation drift <= eps_rep.
  3. Lock the selected gamma.
  4. Evaluate that locked candidate on the disjoint confirmation pool.

No gradient-based training is performed.
"""

import argparse
import copy
import csv
import gc
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scipy.optimize import linear_sum_assignment

from otbli import load_model
from otbli.arch import get_layers
from otbli.atomize import (
    build_barycentric_block,
    build_interpolated_block,
    build_duplicate_block,
    mlp_atoms,
)
from otbli.sinkhorn import pairwise_sq_dists
from otbli.config import OTConfig, DataConfig
from otbli.data import load_or_build_pools
from otbli.insertion import GatedInsertion
from otbli.metrics import batch_losses, batch_kl, bootstrap_ci, paired_t
from otbli.protocol import measure_drift1


DEFAULT_GRID = (
    0.0,
    1e-4, 3e-4,
    1e-3, 3e-3,
    1e-2, 3e-2,
    5e-2, 7.5e-2,
    1e-1, 1.25e-1, 1.5e-1,
    2e-1, 3e-1, 5e-1,
)


def _model_dtype(model):
    return next(model.parameters()).dtype


def _freeze(module):
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)
    return module


def _free(device):
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()


@torch.no_grad()
def build_random_native_block(model, template, seed: int):
    """Random block using the Hugging Face model's native weight initializer."""
    blk = copy.deepcopy(template)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if not hasattr(model, "_init_weights"):
        raise RuntimeError(
            f"{type(model).__name__} does not expose _init_weights; "
            "cannot construct the native-random baseline safely."
        )
    blk.apply(model._init_weights)
    return _freeze(blk)


@torch.no_grad()
def build_hard_ot_midpoint(layer_a, layer_b, tau: float, device: str,
                           gauge_fix: bool = True):
    """
    One-to-one assignment of the exchangeable MLP units under squared Euclidean
    atom cost, followed by matched displacement interpolation.

    All non-exchangeable/single-atom parameter families use the same linear
    interpolation as the barycenter construction, so the difference isolates
    the MLP transport rule.
    """
    X = mlp_atoms(layer_a, gauge_fix=gauge_fix).to(
        device=device, dtype=torch.float32
    )
    Y = mlp_atoms(layer_b, gauge_fix=gauge_fix).to(
        device=device, dtype=torch.float32
    )

    if X.shape[0] != Y.shape[0]:
        raise RuntimeError(
            "hard_ot_midpoint currently requires equal MLP widths: "
            f"{X.shape[0]} vs {Y.shape[0]}"
        )

    C = pairwise_sq_dists(X, Y)
    rows, cols = linear_sum_assignment(C.detach().cpu().numpy())

    # scipy returns rows in sorted order for a square assignment, but make the
    # ordering explicit so Z[r] corresponds to source atom X[r].
    perm = np.empty(X.shape[0], dtype=np.int64)
    perm[rows] = cols
    perm_t = torch.as_tensor(perm, device=device, dtype=torch.long)

    Z = (1.0 - tau) * X + tau * Y[perm_t]

    blk = build_interpolated_block(layer_a, layer_b, tau, mlp_Z=Z)
    return blk


@torch.no_grad()
def build_method_block(model, i, method, ot_cfg, device, random_seed=None):
    layers = get_layers(model)
    a, b = layers[i], layers[i + 1]
    dt = _model_dtype(model)

    if method == "barycenter":
        blk = build_barycentric_block(
            a, b,
            ot_cfg.tau,
            ot_cfg.eta,
            ot_cfg.n_alt_rounds,
            ot_cfg.n_sinkhorn_iters,
            device,
            ot_dtype=getattr(torch, ot_cfg.ot_dtype),
            gauge_fix=ot_cfg.gauge_fix,
            verbose=False,
        )
    elif method == "hard_ot_midpoint":
        blk = build_hard_ot_midpoint(
            a, b, ot_cfg.tau, device, gauge_fix=ot_cfg.gauge_fix
        )
    elif method == "naive_average":
        blk = build_interpolated_block(a, b, ot_cfg.tau)
    elif method == "copy_prev":
        blk = build_duplicate_block(a)
    elif method == "copy_next":
        blk = build_duplicate_block(b)
    elif method == "random_native":
        if random_seed is None:
            raise ValueError("random_seed is required for random_native")
        blk = build_random_native_block(model, a, random_seed)
    else:
        raise ValueError(f"unknown method: {method}")

    return blk.to(device=device, dtype=dt)


def _method_label(method, random_seed=None):
    if method == "random_native":
        return f"random_native_s{random_seed}"
    return method


@torch.no_grad()
def evaluate_method(
    model,
    i,
    method,
    random_seed,
    sel,
    conf,
    base_sel,
    base_conf,
    args,
    ot_cfg,
):
    label = _method_label(method, random_seed)
    print("\n" + "=" * 88)
    print(f"METHOD: {label}")
    print("=" * 88, flush=True)

    t0 = time.time()
    blk = build_method_block(
        model, i, method, ot_cfg, args.device, random_seed=random_seed
    )
    ins = GatedInsertion(model, i + 1, blk)

    drift1 = measure_drift1(
        model, ins, sel, args.device, min(args.drift_batches, len(sel))
    )
    L0 = float(base_sel.mean())

    records = []
    paired_sel = {}

    for gamma in args.gamma_grid:
        gamma = float(gamma)
        if gamma == 0.0:
            rec = dict(
                gamma=0.0,
                loss=L0,
                dL=0.0,
                ci_lo=0.0,
                ci_hi=0.0,
                KL=0.0,
                rep=0.0,
                stable=True,
            )
            records.append(rec)
            continue

        ins.set_gamma(gamma)
        losses = batch_losses(model, sel, args.device)
        ins.set_gamma(0.0)

        d = losses - base_sel
        paired_sel[gamma] = d
        m = float(losses.mean())
        dL = float(d.mean())
        ci_lo, ci_hi = bootstrap_ci(
            d, args.n_boot_sel, seed=1000 + int(round(gamma * 1e6))
        )
        rep = float(gamma * gamma * drift1)

        # Compute KL only for candidates that can still beat gamma=0 in loss
        # and satisfy representation stability. Otherwise KL is unnecessary
        # for the minimization.
        if dL <= 0.0 and rep <= args.eps_rep:
            ins.set_gamma(gamma)
            KL = float(
                batch_kl(
                    model,
                    ins,
                    sel[: min(args.kl_batches, len(sel))],
                    args.device,
                )
            )
            ins.set_gamma(0.0)
        else:
            KL = float("nan")

        stable = bool(
            dL <= 0.0
            and rep <= args.eps_rep
            and np.isfinite(KL)
            and KL <= args.eps_kl
        )

        rec = dict(
            gamma=gamma,
            loss=m,
            dL=dL,
            ci_lo=float(ci_lo),
            ci_hi=float(ci_hi),
            KL=KL,
            rep=rep,
            stable=stable,
        )
        records.append(rec)

        kl_txt = f"{KL:.3e}" if np.isfinite(KL) else "skip"
        print(
            f"  gamma={gamma:<7g} dL={dL:+.6e} "
            f"CI=[{ci_lo:+.2e},{ci_hi:+.2e}] "
            f"KL={kl_txt} rep={rep:.3e} "
            f"{'stable' if stable else ''}",
            flush=True,
        )

    # gamma=0 is always feasible fallback. Among positive stable gates, pick
    # minimum selection loss; conservative smallest-gamma tie-break.
    positive = [r for r in records if r["gamma"] > 0.0 and r["stable"]]
    if positive:
        positive.sort(key=lambda r: (r["loss"], r["gamma"]))
        best_pos = positive[0]
        if best_pos["loss"] < L0:
            selected = best_pos
        else:
            selected = next(r for r in records if r["gamma"] == 0.0)
    else:
        selected = next(r for r in records if r["gamma"] == 0.0)

    gamma_star = float(selected["gamma"])
    print(
        f"  ==> selected gamma={gamma_star:g}, "
        f"selection dL={selected['dL']:+.6e}",
        flush=True,
    )

    # Locked out-of-sample evaluation on the disjoint confirmation pool.
    if gamma_star == 0.0:
        cand_conf = base_conf.copy()
        conf_kl = 0.0
        conf_rep = 0.0
    else:
        ins.set_gamma(gamma_star)
        cand_conf = batch_losses(model, conf, args.device)
        conf_kl = float(
            batch_kl(
                model,
                ins,
                conf[: min(args.kl_batches_conf, len(conf))],
                args.device,
            )
        )
        ins.set_gamma(0.0)
        conf_rep = float(gamma_star * gamma_star * drift1)

    d_conf = cand_conf - base_conf
    conf_mean = float(d_conf.mean())
    conf_lo, conf_hi = bootstrap_ci(
        d_conf, args.n_boot_conf, seed=99
    )
    t_stat, p_one = paired_t(d_conf)
    confirmed = bool(
        gamma_star > 0.0
        and conf_mean < 0.0
        and conf_hi < 0.0
        and conf_kl <= args.eps_kl
        and conf_rep <= args.eps_rep
    )

    base_conf_loss = float(base_conf.mean())
    cand_conf_loss = float(cand_conf.mean())

    print("  ---- locked confirmation ----")
    print(f"  dL_conf       = {conf_mean:+.8e}")
    print(f"  CI95_conf     = [{conf_lo:+.8e}, {conf_hi:+.8e}]")
    print(f"  one-sided p   = {p_one:.6g}")
    print(f"  KL_conf       = {conf_kl:.8e}")
    print(f"  rep_conf      = {conf_rep:.8e}")
    print(
        f"  verdict       = "
        f"{'CONFIRMED' if confirmed else 'NOT CONFIRMED'}",
        flush=True,
    )

    ins.remove()
    del ins, blk
    _free(args.device)

    return {
        "method": label,
        "base_method": method,
        "random_seed": random_seed,
        "position": i,
        "pair": [i + 1, i + 2],
        "selected_gamma": gamma_star,
        "selection_dL": float(selected["dL"]),
        "selection_ci": [
            float(selected["ci_lo"]),
            float(selected["ci_hi"]),
        ],
        "selection_KL": (
            float(selected["KL"])
            if np.isfinite(selected["KL"])
            else None
        ),
        "selection_rep": float(selected["rep"]),
        "confirmation_base_loss": base_conf_loss,
        "confirmation_candidate_loss": cand_conf_loss,
        "confirmation_dL": conf_mean,
        "confirmation_ci": [float(conf_lo), float(conf_hi)],
        "confirmation_t": float(t_stat),
        "confirmation_p_one_sided": float(p_one),
        "confirmation_KL": conf_kl,
        "confirmation_rep": conf_rep,
        "base_ppl": math.exp(base_conf_loss),
        "candidate_ppl": math.exp(cand_conf_loss),
        "confirmed": confirmed,
        "records": records,
        "paired_confirmation": d_conf.tolist(),
        "wallclock_s": time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model",
        default="openai-community/gpt2-large",
    )
    ap.add_argument(
        "--position",
        type=int,
        default=11,
        help="0-indexed pair; 11 corresponds to paper pair (F12,F13)",
    )
    ap.add_argument("--pools", required=True)
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--methods",
        nargs="+",
        default=[
            "barycenter",
            "hard_ot_midpoint",
            "naive_average",
            "copy_prev",
            "copy_next",
            "random_native",
        ],
    )
    ap.add_argument(
        "--random-seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
    )
    ap.add_argument(
        "--gamma-grid",
        nargs="+",
        type=float,
        default=list(DEFAULT_GRID),
    )
    ap.add_argument("--n-boot-sel", type=int, default=5000)
    ap.add_argument("--n-boot-conf", type=int, default=10000)
    ap.add_argument("--kl-batches", type=int, default=25)
    ap.add_argument("--kl-batches-conf", type=int, default=40)
    ap.add_argument("--drift-batches", type=int, default=4)
    ap.add_argument("--eps-kl", type=float, default=0.05)
    ap.add_argument("--eps-rep", type=float, default=0.05)

    # OT construction settings fixed to the current paper values.
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--alt-rounds", type=int, default=25)
    ap.add_argument("--sinkhorn-iters", type=int, default=80)
    ap.add_argument("--ot-dtype", default="float32")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("=" * 88)
    print("GPT-2 FIXED-POSITION BASELINE COMPARISON")
    print("=" * 88)
    print(f"model            : {args.model}")
    print(
        f"fixed position   : {args.position} "
        f"(pair F{args.position+1},F{args.position+2})"
    )
    print(f"methods          : {args.methods}")
    print(f"random seeds     : {args.random_seeds}")
    print(f"gamma grid       : {tuple(args.gamma_grid)}")
    print("selection pool   : gamma tuning only")
    print("confirmation     : locked candidates only")
    print("training         : NONE")
    print("=" * 88, flush=True)

    model, tok = load_model(args.model, device=args.device)

    data_cfg = DataConfig()
    sel, conf = load_or_build_pools(tok, data_cfg, args.pools)
    print(
        f"pools: selection={tuple(sel.shape)}, "
        f"confirmation={tuple(conf.shape)}",
        flush=True,
    )

    base_sel = batch_losses(model, sel, args.device)
    base_conf = batch_losses(model, conf, args.device)

    print(
        f"base selection loss   = {base_sel.mean():.8f}, "
        f"PPL={math.exp(float(base_sel.mean())):.6f}"
    )
    print(
        f"base confirmation loss= {base_conf.mean():.8f}, "
        f"PPL={math.exp(float(base_conf.mean())):.6f}",
        flush=True,
    )

    ot_cfg = OTConfig()
    ot_cfg.tau = args.tau
    ot_cfg.eta = args.eta
    ot_cfg.n_alt_rounds = args.alt_rounds
    ot_cfg.n_sinkhorn_iters = args.sinkhorn_iters
    ot_cfg.ot_dtype = args.ot_dtype

    results = []

    for method in args.methods:
        if method == "random_native":
            for seed in args.random_seeds:
                results.append(
                    evaluate_method(
                        model,
                        args.position,
                        method,
                        seed,
                        sel,
                        conf,
                        base_sel,
                        base_conf,
                        args,
                        ot_cfg,
                    )
                )
        else:
            results.append(
                evaluate_method(
                    model,
                    args.position,
                    method,
                    None,
                    sel,
                    conf,
                    base_sel,
                    base_conf,
                    args,
                    ot_cfg,
                )
            )

    # Sort by out-of-sample confirmation loss change.
    results_sorted = sorted(
        results,
        key=lambda r: r["confirmation_dL"],
    )

    json_path = os.path.join(
        args.out,
        "gpt2_fixed_position_baseline_comparison.json",
    )
    with open(json_path, "w") as f:
        json.dump(
            {
                "model": args.model,
                "position": args.position,
                "pair": [args.position + 1, args.position + 2],
                "gamma_grid": args.gamma_grid,
                "results": results_sorted,
            },
            f,
            indent=2,
        )

    csv_path = os.path.join(
        args.out,
        "gpt2_fixed_position_baseline_comparison.csv",
    )
    fields = [
        "method",
        "selected_gamma",
        "selection_dL",
        "selection_ci_lo",
        "selection_ci_hi",
        "selection_KL",
        "selection_rep",
        "confirmation_dL",
        "confirmation_ci_lo",
        "confirmation_ci_hi",
        "confirmation_p_one_sided",
        "confirmation_KL",
        "confirmation_rep",
        "base_ppl",
        "candidate_ppl",
        "confirmed",
        "wallclock_s",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results_sorted:
            w.writerow(
                {
                    "method": r["method"],
                    "selected_gamma": r["selected_gamma"],
                    "selection_dL": r["selection_dL"],
                    "selection_ci_lo": r["selection_ci"][0],
                    "selection_ci_hi": r["selection_ci"][1],
                    "selection_KL": r["selection_KL"],
                    "selection_rep": r["selection_rep"],
                    "confirmation_dL": r["confirmation_dL"],
                    "confirmation_ci_lo": r["confirmation_ci"][0],
                    "confirmation_ci_hi": r["confirmation_ci"][1],
                    "confirmation_p_one_sided": r[
                        "confirmation_p_one_sided"
                    ],
                    "confirmation_KL": r["confirmation_KL"],
                    "confirmation_rep": r["confirmation_rep"],
                    "base_ppl": r["base_ppl"],
                    "candidate_ppl": r["candidate_ppl"],
                    "confirmed": r["confirmed"],
                    "wallclock_s": r["wallclock_s"],
                }
            )

    print("\n" + "=" * 88)
    print("FINAL RANKING BY LOCKED CONFIRMATION dL")
    print("=" * 88)
    for r in results_sorted:
        print(
            f"{r['method']:<24s} "
            f"gamma={r['selected_gamma']:<7g} "
            f"dL_sel={r['selection_dL']:+.3e} "
            f"dL_conf={r['confirmation_dL']:+.3e} "
            f"CI=[{r['confirmation_ci'][0]:+.2e},"
            f"{r['confirmation_ci'][1]:+.2e}] "
            f"{'CONFIRMED' if r['confirmed'] else ''}"
        )
    print("=" * 88)
    print(f"JSON -> {json_path}")
    print(f"CSV  -> {csv_path}")


if __name__ == "__main__":
    main()
