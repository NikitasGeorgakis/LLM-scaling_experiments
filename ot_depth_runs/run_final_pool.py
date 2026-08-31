#!/usr/bin/env python3
"""Single locked evaluation of saved candidates on one untouched pool.

E1: python run_final_pool.py --model-key gpt2-large --pool pools/C_final.jsonl \
        --n-batches 40 --candidates runs/locked_gpt2/*.pt runs/e3_*/locked/*.pt \
        --pairwise copy_next,hard_ot,barycenter,naive --out runs/e1_final
E7: same script with --pool pools/F_openwebtext.jsonl (and again on a fresh
    Pile pool) to compare in- vs out-of-distribution behaviour.

Decision rule (pre-registered): a candidate is confirmed when the 95%
paired batch-bootstrap interval (10,000 resamples) lies below zero and the
stability constraints hold; one-sided t p-values are reported raw and
Holm-adjusted across the primary candidates.
"""
import argparse
import glob
import itertools
import json
import os
import time

import numpy as np

import otdepth as od


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--n-batches", type=int, default=40)
    ap.add_argument("--candidates", nargs="+", required=True)
    ap.add_argument("--pairwise", default="",
                    help="comma list of constructions to compare head-to-head")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--kl-probe-batches", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    paths = sorted(itertools.chain.from_iterable(glob.glob(p)
                                                 for p in a.candidates))
    model, tok = od.load_model(a.model_key, a.revision, a.dtype, a.device)
    pool = od.pack_pool(a.pool, tok, a.n_batches)
    base_losses, probes = od.base_pass(model, pool, a.device,
                                       a.kl_probe_batches)
    L0 = float(base_losses.mean())
    print(f"pool={a.pool} sha256={od.sha256_file(a.pool)[:16]} "
          f"L0={L0:.6f} ppl0={np.exp(L0):.5f}")

    results, loss_mat, names = [], {}, []
    for path in paths:
        meta, cand = od.load_locked(path, model)
        name = meta["construction"]
        if name in loss_mat:
            name = f"{name}@{meta['i']+1}-{meta['i']+2}"
        g = meta["gamma"]
        rec, diffs = od.eval_candidate(model, pool, base_losses, probes,
                                       meta["i"], cand, [0.0, g], a.device,
                                       a.kl_probe_batches)
        del cand
        d = diffs[1]
        ci = od.boot_ci(d, a.boot, seed=11)
        t, p = od.one_sided_t(d)
        r = rec[f"{g}"]
        confirmed = bool(ci[1] < 0 and r["feasible"])
        res = dict(candidate=name, file=path, i=meta["i"],
                   pair=f"{meta['i']+1}-{meta['i']+2}", gamma=g,
                   dL=float(d.mean()), ci_lo=ci[0], ci_hi=ci[1],
                   t=t, p_one_sided=p, KL=r["KL"], rep=r["rep"],
                   ppl_improvement_pct=float((1 - np.exp(d.mean())) * 100),
                   confirmed=confirmed)
        results.append(res)
        loss_mat[name] = base_losses + d
        names.append(name)
        print(f"  {name:<14} gamma={g:<5g} dL={d.mean():+.5e} "
              f"CI=[{ci[0]:+.4e},{ci[1]:+.4e}] p={p:.4f} conf={confirmed}")

    padj = holm(np.array([r["p_one_sided"] for r in results]))
    for r, pa in zip(results, padj):
        r["p_holm"] = float(pa)

    pairwise = []
    plist = [x for x in a.pairwise.split(",") if x]
    for m1, m2 in itertools.combinations(plist, 2):
        if m1 not in loss_mat or m2 not in loss_mat:
            continue
        d = loss_mat[m1] - loss_mat[m2]          # negative -> m1 better
        ci = od.boot_ci(d, a.boot, seed=23)
        t, p = od.one_sided_t(d)
        pairwise.append(dict(a=m1, b=m2, dL_a_minus_b=float(d.mean()),
                             ci_lo=ci[0], ci_hi=ci[1], t=t, p_one_sided=p))
        print(f"  {m1} vs {m2}: dL={d.mean():+.5e} "
              f"CI=[{ci[0]:+.4e},{ci[1]:+.4e}] p={p:.4f}")

    out = dict(model=od.MODELS.get(a.model_key, a.model_key),
               revision=a.revision, pool=a.pool,
               pool_sha256=od.sha256_file(a.pool), n_batches=a.n_batches,
               L0=L0, ppl0=float(np.exp(L0)),
               base_losses=base_losses.tolist(),
               results=results, pairwise=pairwise,
               per_candidate_losses={k: v.tolist()
                                     for k, v in loss_mat.items()},
               wallclock_s=time.time() - t0)
    json.dump(out, open(os.path.join(a.out, "final.json"), "w"), indent=1)
    import csv
    with open(os.path.join(a.out, "final.csv"), "w", newline="") as f:
        keys = [k for k in results[0] if k != "file"]
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        [w.writerow(r) for r in results]
    print(f"done in {out['wallclock_s']:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
