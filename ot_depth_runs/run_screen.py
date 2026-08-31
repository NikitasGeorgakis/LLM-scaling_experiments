#!/usr/bin/env python3
"""Corrected all-position small-gate screen, with optional refinement at the
best position and one locked disjoint confirmation.

Covers:
  E2: per Pythia model      -> --model-key pythia-410m --construction barycenter
                               --refine --confirm-pool pools/B_conf.jsonl
  E3: per construction      -> --model-key gpt2-large --construction copy_next
                               --refine   (no --confirm-pool: final test is E1)
  E4: per trajectory step   -> --model-key pythia-1.4b --revision step512 ...
  E5: extra selection pools -> same command on pools/D_sel.jsonl, E_sel.jsonl

Outputs in --out: screen.csv, screen.npz (per-batch diffs for E5 max-t),
summary.json, and if --refine: locked/<construction>.pt (+ confirm.json).
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import otdepth as od


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--construction", default="barycenter",
                    choices=["barycenter", "hard_ot", "copy_next", "copy_prev",
                             "naive", "random"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--n-batches", type=int, default=25)
    ap.add_argument("--grid", default="small", choices=["small", "full"])
    ap.add_argument("--positions", default="all",
                    help="'all' or comma list of 0-indexed left blocks")
    ap.add_argument("--refine", action="store_true")
    ap.add_argument("--refine-at", type=int, default=None,
                    help="force refinement position (0-indexed left block)")
    ap.add_argument("--confirm-pool", default=None)
    ap.add_argument("--confirm-batches", type=int, default=40)
    ap.add_argument("--boot-select", type=int, default=2000)
    ap.add_argument("--boot-confirm", type=int, default=10000)
    ap.add_argument("--kl-probe-batches", type=int, default=1)
    ap.add_argument("--bary-eta", type=float, default=0.05)
    ap.add_argument("--bary-rounds", type=int, default=25)
    ap.add_argument("--bary-iters", type=int, default=80)
    ap.add_argument("--cost-norm", default="mean", choices=["mean", "none"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--skip-exact-check", action="store_true")
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def main():
    a = parse_args()
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    grid = od.SMALL_GRID if a.grid == "small" else od.FULL_GRID
    bary = dict(eta=a.bary_eta, rounds=a.bary_rounds,
                sinkhorn_iters=a.bary_iters, cost_norm=a.cost_norm)

    model, tok = od.load_model(a.model_key, a.revision, a.dtype, a.device)
    n_blocks = len(od.get_blocks(model))
    pool = od.pack_pool(a.pool, tok, a.n_batches)
    base_losses, probes = od.base_pass(model, pool, a.device,
                                       a.kl_probe_batches)
    L0 = float(base_losses.mean())
    print(f"{a.model_key}{'@'+a.revision if a.revision else ''} "
          f"blocks={n_blocks} L0_sel={L0:.6f} pool={a.pool} "
          f"sha256={od.sha256_file(a.pool)[:16]}")

    if a.positions == "all":
        positions = list(range(n_blocks - 1))
    else:
        positions = [int(x) for x in a.positions.split(",")]

    if not a.skip_exact_check:
        cand0 = od.build_candidate(model, positions[0], "copy_next")
        ok, mx = od.check_exact_recovery(model, pool, positions[0], cand0,
                                         a.device)
        print(f"exact recovery at gamma=0: {ok} (max|dlogit|={mx:.3g})")
        del cand0

    n_gam = len(grid)
    diffs_all = np.zeros((len(positions), n_gam, a.n_batches))
    feas_all = np.zeros((len(positions), n_gam), dtype=bool)
    sel_idx = np.zeros(len(positions), dtype=np.int64)
    rows, per_pos_records = [], {}
    for pi, i in enumerate(positions):
        cand = od.build_candidate(model, i, a.construction, a.tau, a.seed,
                                  bary, verbose=True)
        rec, diffs = od.eval_candidate(model, pool, base_losses, probes, i,
                                       cand, grid, a.device,
                                       a.kl_probe_batches)
        del cand
        torch.cuda.empty_cache() if a.device == "cuda" else None
        diffs_all[pi] = diffs
        feas_all[pi] = [rec[f"{g}"]["feasible"] for g in grid]
        g_star, dL = od.select_gamma(rec)
        gi = grid.index(g_star)
        sel_idx[pi] = gi
        ci = (od.boot_ci(diffs[gi], a.boot_select, seed=1000 + i)
              if g_star > 0 else [0.0, 0.0])
        r = rec[f"{g_star}"]
        rows.append(dict(i=i, pair=f"{i+1}-{i+2}", gamma_star=g_star,
                         dL_sel=dL, ci_lo=ci[0], ci_hi=ci[1],
                         KL=r["KL"], rep=r["rep"]))
        per_pos_records[str(i)] = rec
        print(f"  pos {i:>2} (F{i+1},F{i+2}): gamma*={g_star:<7g} "
              f"dL={dL:+.4e}  CI=[{ci[0]:+.3e},{ci[1]:+.3e}]")

    # ---- persist screen ------------------------------------------------
    import csv
    with open(os.path.join(a.out, "screen.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        [w.writerow(r) for r in rows]
    np.savez_compressed(
        os.path.join(a.out, "screen.npz"), diffs=diffs_all,
        feasible=feas_all, sel_idx=sel_idx, gammas=np.array(grid),
        positions=np.array(positions),
        meta=json.dumps(vars(a) | {"L0_sel": L0, "n_blocks": n_blocks}))

    pos_sel = [(r["dL_sel"], r["i"]) for r in rows if r["gamma_star"] > 0]
    best_i = min(pos_sel)[1] if pos_sel else None
    n_pos_gates = sum(1 for r in rows if r["gamma_star"] > 0)
    print(f"positive gates: {n_pos_gates}/{len(rows)}; best position: {best_i}")

    summary = dict(model=od.MODELS.get(a.model_key, a.model_key),
                   revision=a.revision, blocks=n_blocks, L0_sel=L0,
                   base_sel_losses=base_losses.tolist(),
                   construction=a.construction, grid=grid,
                   screen=rows, best_i=best_i,
                   pool=a.pool, pool_sha256=od.sha256_file(a.pool))

    # ---- refinement at the pre-selected position ----------------------
    if a.refine and (best_i is not None or a.refine_at is not None):
        i = a.refine_at if a.refine_at is not None else best_i
        rgrid = sorted(set([0.0] + od.REFINE_GRID
                           + [rows[positions.index(i)]["gamma_star"]]))
        cand = od.build_candidate(model, i, a.construction, a.tau, a.seed,
                                  bary)
        rec, diffs = od.eval_candidate(model, pool, base_losses, probes, i,
                                       cand, rgrid, a.device,
                                       a.kl_probe_batches)
        g_ref, dL_ref = od.select_gamma(rec)
        gi = rgrid.index(g_ref)
        ci = od.boot_ci(diffs[gi], a.boot_select, seed=77)
        r = rec[f"{g_ref}"]
        print(f"refined at pos {i}: gamma**={g_ref} dL={dL_ref:+.4e} "
              f"CI=[{ci[0]:+.3e},{ci[1]:+.3e}] KL={r['KL']:.3e} "
              f"rep={r['rep']:.3e}")
        lpath = os.path.join(a.out, "locked", f"{a.construction}.pt")
        if g_ref == 0.0:
            print("refined gate fell back to 0; skipping lock and "
                  "confirmation")
            summary["refined"] = dict(i=i, gamma=0.0, dL_sel=0.0)
            a.confirm_pool = None
        else:
            od.save_locked(lpath, a.model_key, a.revision, i, a.construction,
                           g_ref, a.tau, a.seed, cand,
                           extra=dict(dL_sel=dL_ref, ci_sel=ci, KL=r["KL"],
                                      rep=r["rep"], refine_grid=rgrid,
                                      screen_pool=a.pool,
                                      bary=bary
                                      if a.construction == "barycenter"
                                      else None))
            summary["refined"] = dict(i=i, gamma=g_ref, dL_sel=dL_ref, ci=ci,
                                      KL=r["KL"], rep=r["rep"], locked=lpath)

        # ---- locked disjoint confirmation -----------------------------
        if a.confirm_pool:
            cpool = od.pack_pool(a.confirm_pool, tok, a.confirm_batches)
            cbase, cprobes = od.base_pass(model, cpool, a.device,
                                          a.kl_probe_batches)
            crec, cdiffs = od.eval_candidate(model, cpool, cbase, cprobes, i,
                                             cand, [0.0, g_ref], a.device,
                                             a.kl_probe_batches)
            d = cdiffs[1]
            cci = od.boot_ci(d, a.boot_confirm, seed=7)
            t, p = od.one_sided_t(d)
            cr = crec[f"{g_ref}"]
            confirmed = bool(cci[1] < 0 and cr["feasible"])
            conf = dict(pool=a.confirm_pool,
                        pool_sha256=od.sha256_file(a.confirm_pool),
                        n_batches=a.confirm_batches, i=i, gamma=g_ref,
                        L_conf_0=float(cbase.mean()),
                        L_conf_g=float(cbase.mean() + d.mean()),
                        dL_conf=float(d.mean()), ci=cci, t=t, p_one_sided=p,
                        KL=cr["KL"], rep=cr["rep"],
                        ppl0=float(np.exp(cbase.mean())),
                        ppl1=float(np.exp(cbase.mean() + d.mean())),
                        confirmed=confirmed,
                        per_batch=d.tolist())
            json.dump(conf, open(os.path.join(a.out, "confirm.json"), "w"),
                      indent=1)
            print(f"CONFIRMATION: dL={d.mean():+.5e} CI=[{cci[0]:+.4e},"
                  f"{cci[1]:+.4e}] p={p:.4f} confirmed={confirmed}")
            summary["confirmation"] = {k: v for k, v in conf.items()
                                       if k != "per_batch"}
        del cand

    summary["wallclock_s"] = time.time() - t0
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"),
              indent=1)
    print(f"done in {summary['wallclock_s']:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
