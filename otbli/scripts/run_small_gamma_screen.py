#!/usr/bin/env python3
"""Corrected all-position small-gamma screen for OT-BLI.

This script implements the revised screening protocol proposed after the
initial experiments:

  * every valid insertion position is screened with a small-gamma grid;
  * C_eff is NOT used to select gamma;
  * no materiality margin delta is used;
  * the selected gate minimizes validation loss subject only to the
    pre-specified output-KL and representation-drift constraints;
  * gamma=0 is always included and is the exact safe fallback;
  * the confirmation pool is loaded/cached but never evaluated by this script;
  * no model parameter is updated (forward-only evaluation).

For compute efficiency, output KL is evaluated lazily: for a positive gamma it
is computed only if its paired validation loss is no worse than gamma=0 and its
representation drift already satisfies the drift constraint.  This does not
change the selected gamma, because gamma=0 is always available; a candidate
with larger validation loss cannot be the minimizer.

Run from the repository root, e.g.

    python scripts/run_small_gamma_screen.py \
        --model EleutherAI/pythia-410m --device cuda \
        --out results/small_gamma

The output JSON contains the complete per-position/per-gamma records.  A CSV
contains one row per insertion position for quick ranking.
"""

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from otbli import load_model
from otbli.arch import get_layers
from otbli.atomize import _detect_mlp_type
from otbli.config import OTConfig, DataConfig, ProtocolConfig
from otbli.data import load_or_build_pools
from otbli.metrics import batch_losses, batch_kl, bootstrap_ci
from otbli.protocol import make_insertion, measure_drift1


DEFAULT_SMALL_GRID = (
    0.0,
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
    3e-2,
    1e-1,
)


def _free(device):
    import gc
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()


def _json_dump(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
    os.replace(tmp, path)


def _best_stable_record(records, eps_kl, eps_rep):
    """Loss minimizer among stability-feasible candidates; gamma=0 is safe.

    Positive-gamma records for which KL was not computed are noncompetitive
    (their selection loss exceeded the base loss), so they are excluded.
    """
    candidates = []
    for rec in records:
        g = rec["gamma"]
        if g == 0.0:
            candidates.append(rec)
            continue
        if rec["KL"] is None:
            continue
        if rec["KL"] <= eps_kl and rec["rep"] <= eps_rep:
            candidates.append(rec)

    if not candidates:
        return records[0]

    best_loss = min(r["loss"] for r in candidates)
    # Conservative tie-break: smallest gamma among numerically equal minima.
    tied = [r for r in candidates if abs(r["loss"] - best_loss) <= 1e-12]
    return min(tied, key=lambda r: r["gamma"])


@torch.no_grad()
def screen_model(args):
    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)

    ot_cfg = OTConfig(
        tau=args.tau,
        eta=args.eta,
        n_alt_rounds=args.alt_rounds,
        n_sinkhorn_iters=args.sinkhorn_iters,
        gauge_fix=not args.no_gauge_fix,
        ot_dtype=args.ot_dtype,
    )
    data_cfg = DataConfig()
    if args.dataset:
        data_cfg.dataset = args.dataset
    proto_cfg = ProtocolConfig()

    grid = tuple(sorted(set(float(g) for g in args.gamma_grid)))
    if 0.0 not in grid:
        grid = (0.0,) + grid

    short = args.model.split("/")[-1]
    json_path = os.path.join(args.out, f"small_gamma_{short}.json")
    csv_path = os.path.join(args.out, f"small_gamma_{short}.csv")
    resume_blob = None
    if os.path.exists(json_path) and not args.force:
        try:
            with open(json_path) as f:
                resume_blob = json.load(f)
        except Exception:
            resume_blob = None
        if resume_blob and resume_blob.get("complete", False):
            print(f"[skip] completed result found at {json_path}; use --force to overwrite", flush=True)
            return json_path
        if resume_blob:
            print(f"[resume] incomplete result found at {json_path}; completed positions will be skipped", flush=True)

    print("=" * 88)
    print("CORRECTED SMALL-GAMMA ALL-POSITION SCREEN")
    print("=" * 88)
    print(f"model            : {args.model}")
    print(f"gamma grid       : {grid}")
    print(f"selection batches: {args.screen_batches if args.screen_batches > 0 else 'all'}")
    print(f"KL batches       : {args.kl_batches}")
    print(f"drift batches    : {args.drift_batches}")
    print(f"eps_KL           : {args.eps_kl}")
    print(f"eps_rep          : {args.eps_rep}")
    print("C_eff            : NOT USED")
    print("materiality delta: NOT USED")
    print("confirmation pool: NOT EVALUATED")
    print("=" * 88, flush=True)

    model, tok = load_model(args.model, device=args.device)
    layers = get_layers(model)
    L = len(layers)
    arch = _detect_mlp_type(layers[0])
    n_par = sum(p.numel() for p in model.parameters())
    print(f"layers={L}  parameters={n_par/1e6:.1f}M  MLP family={arch}", flush=True)

    pools_path = args.pools or os.path.join(
        args.pool_dir, f"pools_{short}_seed{data_cfg.seed}.pt"
    )
    sel, _conf = load_or_build_pools(tok, data_cfg, pools_path)
    # The cached blob contains the confirmation tensor, so torch.load materializes it,
    # but this script never indexes, scores, summarizes, or otherwise evaluates it.
    if args.screen_batches > 0:
        if args.screen_batches > len(sel):
            raise ValueError(
                f"--screen-batches={args.screen_batches} exceeds cached selection "
                f"pool size {len(sel)}"
            )
        sel_eval = sel[: args.screen_batches]
    else:
        sel_eval = sel

    base_losses = batch_losses(model, sel_eval, args.device)
    L0 = float(base_losses.mean())
    print(f"base selection loss = {L0:.8f} nats/token  PPL={np.exp(L0):.5f}", flush=True)

    if args.positions:
        positions = list(args.positions)
    else:
        positions = list(range(L - 1))
    bad = [i for i in positions if i < 0 or i >= L - 1]
    if bad:
        raise ValueError(f"invalid 0-indexed insertion positions {bad}; valid range is 0..{L-2}")

    results = []
    if resume_blob and not args.force:
        same_grid = tuple(float(x) for x in resume_blob.get("grid", [])) == grid
        same_eps = (float(resume_blob.get("eps_KL", args.eps_kl)) == float(args.eps_kl) and
                    float(resume_blob.get("eps_rep", args.eps_rep)) == float(args.eps_rep))
        same_batches = int(resume_blob.get("screen_batches", len(sel_eval))) == len(sel_eval)
        if not (same_grid and same_eps and same_batches and resume_blob.get("model") == args.model):
            raise RuntimeError(
                "Existing partial JSON was produced with different model/grid/tolerances/screen-batches. "
                "Use --force or a different --out directory."
            )
        results = list(resume_blob.get("results", []))
    completed_positions = {int(r["i"]) for r in results}

    for pos_idx, i in enumerate(positions, 1):
        if i in completed_positions:
            print(f"[resume] skip completed insertion pair (F{i+1},F{i+2})", flush=True)
            continue
        print("\n" + "-" * 88, flush=True)
        print(
            f"[{pos_idx}/{len(positions)}] insertion pair (F{i+1}, F{i+2}) "
            f"[0-indexed i={i}]",
            flush=True,
        )
        pos_t0 = time.time()

        ins = make_insertion(model, i, ot_cfg, args.device, verbose=args.verbose_barycenter)
        drift1 = measure_drift1(
            model, ins, sel_eval, args.device,
            min(args.drift_batches, len(sel_eval)),
        )
        print(f"D_rep(1) = {drift1:.8e}", flush=True)

        recs = []
        for gamma in grid:
            if gamma == 0.0:
                rec = {
                    "gamma": 0.0,
                    "loss": L0,
                    "dL": 0.0,
                    "ci95": [0.0, 0.0],
                    "rep": 0.0,
                    "KL": 0.0,
                    "KL_evaluated": True,
                    "stable": True,
                    "slope_dL_over_gamma": None,
                }
                recs.append(rec)
                print("  gamma=0          dL=+0.000e+00  KL=0  rep=0  [baseline]", flush=True)
                continue

            ins.set_gamma(gamma)
            losses = batch_losses(model, sel_eval, args.device)
            ins.set_gamma(0.0)
            d = losses - base_losses
            dL = float(d.mean())
            ci = bootstrap_ci(d, args.n_boot, seed=args.seed + 1000 * i + int(gamma * 1e8) % 997)
            rep = float(gamma * gamma * drift1)

            # Lazy KL is selection-equivalent because gamma=0 is in the grid.
            # A positive-gamma candidate with dL>0 cannot minimize validation loss.
            need_kl = args.kl_all or (dL <= 0.0 and rep <= args.eps_rep)
            kl = None
            stable = None
            if need_kl:
                ins.set_gamma(gamma)
                kl_pool = sel_eval[: min(args.kl_batches, len(sel_eval))]
                kl = float(batch_kl(model, ins, kl_pool, args.device))
                ins.set_gamma(0.0)
                stable = bool(kl <= args.eps_kl and rep <= args.eps_rep)

            rec = {
                "gamma": float(gamma),
                "loss": float(losses.mean()),
                "dL": dL,
                "ci95": [float(ci[0]), float(ci[1])],
                "rep": rep,
                "KL": kl,
                "KL_evaluated": bool(need_kl),
                "stable": stable,
                "slope_dL_over_gamma": float(dL / gamma),
            }
            recs.append(rec)
            kl_txt = "skip" if kl is None else f"{kl:.3e}"
            st_txt = "n/a" if stable is None else ("stable" if stable else "unstable")
            print(
                f"  gamma={gamma:<10g} dL={dL:+.3e}  "
                f"CI=[{ci[0]:+.2e},{ci[1]:+.2e}]  KL={kl_txt:<10s}  "
                f"rep={rep:.3e}  {st_txt}",
                flush=True,
            )

        ins.remove()
        del ins
        _free(args.device)

        best = _best_stable_record(recs, args.eps_kl, args.eps_rep)
        positive = [r for r in recs if r["gamma"] > 0.0]
        slope_rec = min(positive, key=lambda r: r["slope_dL_over_gamma"]) if positive else None

        row = {
            "i": i,
            "pair": [i + 1, i + 2],
            "drift1": drift1,
            "best_gamma": float(best["gamma"]),
            "best_loss": float(best["loss"]),
            "best_dL": float(best["dL"]),
            "best_ci95": best["ci95"],
            "best_KL": best["KL"],
            "best_rep": float(best["rep"]),
            "min_slope_gamma": None if slope_rec is None else float(slope_rec["gamma"]),
            "min_slope": None if slope_rec is None else float(slope_rec["slope_dL_over_gamma"]),
            "records": recs,
            "wallclock_s": time.time() - pos_t0,
        }
        results.append(row)
        print(
            f"  ==> best stable gamma={best['gamma']:g}, dL={best['dL']:+.3e}; "
            f"small-gamma score min(dL/gamma)="
            f"{row['min_slope'] if row['min_slope'] is not None else float('nan'):+.3e}",
            flush=True,
        )

        # Resume safety: write a valid partial JSON after every position.
        partial = {
            "model": args.model,
            "arch": arch,
            "layers": L,
            "params": n_par,
            "base_loss": L0,
            "grid": list(grid),
            "eps_KL": args.eps_kl,
            "eps_rep": args.eps_rep,
            "screen_batches": len(sel_eval),
            "kl_batches": min(args.kl_batches, len(sel_eval)),
            "drift_batches": min(args.drift_batches, len(sel_eval)),
            "pool_path": pools_path,
            "confirmation_pool_evaluated": False,
            "selection_rule": "min validation loss subject to KL/representation stability; gamma=0 included",
            "C_eff_used": False,
            "materiality_delta_used": False,
            "results": results,
            "complete": False,
            "wallclock_s": time.time() - t0,
        }
        _json_dump(partial, json_path)

    # Rank by best observed stable loss change, then by local slope.
    ranked = sorted(results, key=lambda r: (r["best_dL"], r["min_slope"] if r["min_slope"] is not None else float("inf")))
    summary = {
        "model": args.model,
        "arch": arch,
        "layers": L,
        "params": n_par,
        "base_loss": L0,
        "grid": list(grid),
        "eps_KL": args.eps_kl,
        "eps_rep": args.eps_rep,
        "screen_batches": len(sel_eval),
        "kl_batches": min(args.kl_batches, len(sel_eval)),
        "drift_batches": min(args.drift_batches, len(sel_eval)),
        "pool_path": pools_path,
        "confirmation_pool_evaluated": False,
        "selection_rule": "min validation loss subject to KL/representation stability; gamma=0 included",
        "C_eff_used": False,
        "materiality_delta_used": False,
        "best_position": ranked[0]["i"] if ranked else None,
        "best_pair": ranked[0]["pair"] if ranked else None,
        "best_gamma": ranked[0]["best_gamma"] if ranked else 0.0,
        "best_dL": ranked[0]["best_dL"] if ranked else 0.0,
        "results": results,
        "complete": True,
        "wallclock_s": time.time() - t0,
    }
    _json_dump(summary, json_path)

    with open(csv_path, "w", newline="") as f:
        fields = [
            "rank", "i", "pair", "best_gamma", "best_dL", "ci_lo", "ci_hi",
            "best_KL", "best_rep", "D_rep_1", "min_slope_gamma", "min_slope",
            "wallclock_s",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for rank, r in enumerate(ranked, 1):
            w.writerow({
                "rank": rank,
                "i": r["i"],
                "pair": f"F{r['pair'][0]}-F{r['pair'][1]}",
                "best_gamma": r["best_gamma"],
                "best_dL": r["best_dL"],
                "ci_lo": r["best_ci95"][0],
                "ci_hi": r["best_ci95"][1],
                "best_KL": r["best_KL"],
                "best_rep": r["best_rep"],
                "D_rep_1": r["drift1"],
                "min_slope_gamma": r["min_slope_gamma"],
                "min_slope": r["min_slope"],
                "wallclock_s": r["wallclock_s"],
            })

    print("\n" + "=" * 88)
    print(f"DONE {args.model}")
    print(f"JSON -> {json_path}")
    print(f"CSV  -> {csv_path}")
    if ranked:
        r = ranked[0]
        print(
            f"best position: (F{r['pair'][0]},F{r['pair'][1]})  "
            f"gamma={r['best_gamma']:g}  dL={r['best_dL']:+.6e}"
        )
    print(f"wallclock: {time.time()-t0:.1f}s")
    print("=" * 88, flush=True)
    return json_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/small_gamma")
    ap.add_argument("--pool-dir", default="results")
    ap.add_argument("--pools", default=None,
                    help="optional explicit cached pool .pt path")
    ap.add_argument("--dataset", default=None,
                    help="override DataConfig.dataset only if the pool must be built")
    ap.add_argument("--positions", nargs="*", type=int, default=None,
                    help="0-indexed insertion positions; omitted => every valid position")
    ap.add_argument("--gamma-grid", nargs="+", type=float, default=list(DEFAULT_SMALL_GRID))
    ap.add_argument("--screen-batches", type=int, default=25,
                    help="selection batches used at every position; 0 => all cached selection batches")
    ap.add_argument("--kl-batches", type=int, default=25)
    ap.add_argument("--drift-batches", type=int, default=4)
    ap.add_argument("--eps-kl", type=float, default=0.05)
    ap.add_argument("--eps-rep", type=float, default=0.05)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--kl-all", action="store_true",
                    help="compute output KL at every positive gamma (slower); default is lazy KL")

    # Barycenter construction: defaults match the existing project.
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--alt-rounds", type=int, default=25)
    ap.add_argument("--sinkhorn-iters", type=int, default=80)
    ap.add_argument("--ot-dtype", default="float32")
    ap.add_argument("--no-gauge-fix", action="store_true")
    ap.add_argument("--verbose-barycenter", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    screen_model(args)


if __name__ == "__main__":
    main()
