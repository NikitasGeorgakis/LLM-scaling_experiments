#!/usr/bin/env python3
"""Build and save the locked GPT-2 Large candidates at (F12,F13).

gamma for each construction is now determined DYNAMICALLY by the corrected
od.select_gamma() (full J-objective, eq. 3.41/3.43/3.47) -- not a hardcoded
value. The earlier hardcoded set (copy_next=0.5, hard_ot=0.5, barycenter=0.3,
naive=0.3) was traced to the pre-fix loss-only selection heuristic; the
barycenter=0.3 entry specifically DOES have a real, disjoint-pool confirmation
on file (~/otbli/results/gpt2_confirmation_f12_f13_g03/), but recomputing J
from that confirmation's own numbers gives J(0.3) = +7.6e-3 > J(0) = 0 -- it
would never have been selected by the real objective in the first place.
Decision: do not carry it forward. Nothing is locked unless select_gamma()
itself returns gamma* > 0 at this position, so this script now can only ever
produce candidates that are consistent with the corrected screening.
"""
import argparse
import os

import torch

import otdepth as od

I_LOCK = 11    # 0-indexed left block -> pair (F12, F13)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", default="gpt2-large")
    ap.add_argument("--i", type=int, default=I_LOCK)
    ap.add_argument("--constructions", default="copy_next,hard_ot,barycenter,naive")
    ap.add_argument("--pool", required=True,
                    help="selection pool .jsonl -- gamma* is determined on THIS "
                         "pool via od.eval_candidate/od.select_gamma, not assumed")
    ap.add_argument("--grid", default=None,
                    help="comma-separated gamma grid; default od.FULL_GRID")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--kl-probe-batches", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--out", default="runs/locked_gpt2")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    grid = [float(x) for x in args.grid.split(",")] if args.grid else od.FULL_GRID

    model, tok = od.load_model(args.model_key, None, args.dtype, args.device)
    pool = od.pack_pool(args.pool, tok, 25)
    base_losses, probes = od.base_pass(model, pool, args.device, args.kl_probe_batches)

    print(f"determining gamma* dynamically at i={args.i} (F{args.i+1},F{args.i+2}) "
          f"on pool={args.pool}\n")

    locked, skipped = [], []
    for c in args.constructions.split(","):
        cand = od.build_candidate(model, args.i, c, args.tau, seed=0,
                                  bary_kwargs=od.BARY_DEFAULTS if c == "barycenter"
                                  else None, verbose=True)
        rec, diffs = od.eval_candidate(model, pool, base_losses, probes, args.i,
                                       cand, grid, args.device, args.kl_probe_batches)
        gamma, dL = od.select_gamma(rec)
        if gamma == 0.0:
            print(f"  {c:<12s} gamma*=0.0 (safe fallback) -> NOT locked "
                  f"(nothing here clears the pre-registered objective)")
            skipped.append(c)
            del cand
            continue
        path = os.path.join(args.out, f"{c}.pt")
        od.save_locked(path, args.model_key, None, args.i, c, gamma, args.tau, 0,
                       cand, extra={"source": "rebuilt", "dL_at_selection": dL,
                                   "selected_by": "od.select_gamma (eq 3.41/3.43/3.47)"})
        print(f"  {c:<12s} gamma*={gamma} dL={dL:+.4e} -> saved {path}")
        locked.append(c)
        del cand

    print(f"\nlocked: {locked if locked else 'NONE'}   skipped (gamma*=0): {skipped}")
    if not locked:
        print("\nNo construction cleared the selection objective at this position.\n"
              "This is itself the E1 result for gpt2-large @ (F12,F13): report the "
              "null across constructions rather than proceeding to lock/confirm.")


if __name__ == "__main__":
    main()
