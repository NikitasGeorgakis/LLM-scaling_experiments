#!/usr/bin/env python3
"""E5: multiplicity-honest inference and depth-profile stability.

Consumes screen.npz files written by run_screen.py.

  # joint (position x gate) max-t over one screen:
  python run_maxt.py --records runs/e5_gpt2_poolA/screen.npz --out runs/e5

  # profile stability across three pools (A, D, E):
  python run_maxt.py --records runs/e5_gpt2_poolA/screen.npz \
      runs/e5_gpt2_poolD/screen.npz runs/e5_gpt2_poolE/screen.npz --out runs/e5

Method: observed t statistic per feasible (position, gamma>0) cell; the null
distribution of the minimum t is estimated by a centered paired batch
bootstrap; the global p is P(min t* <= observed min), and per-cell single-step
adjusted p is P(min t* <= t_cell). Positive-gamma cells only.
"""
import argparse
import json
import os

import numpy as np


def cell_t(D, mask):
    n = D.shape[-1]
    mean = D.mean(-1)
    sd = D.std(-1, ddof=1)
    se = np.where(sd > 0, sd / np.sqrt(n), np.inf)
    t = np.where(mask, mean / se, np.inf)
    return t, mean


def maxt(D, mask, n_boot, seed):
    """D: [P, G, B]; mask: [P, G] cells under test."""
    t_obs, mean = cell_t(D, mask)
    Dc = D - D.mean(-1, keepdims=True)          # centered null
    rng = np.random.default_rng(seed)
    B = D.shape[-1]
    mins = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.integers(0, B, size=B)
        tb, _ = cell_t(Dc[..., idx], mask)
        mins[k] = tb.min()
    t_min_obs = t_obs.min()
    p_global = float((mins <= t_min_obs).mean())
    p_adj = np.ones_like(t_obs)
    finite = np.isfinite(t_obs)
    p_adj[finite] = [(mins <= tv).mean() for tv in t_obs[finite]]
    return t_obs, mean, p_global, p_adj, t_min_obs


def selected_profile(z):
    """Per-position selected mean dL from a screen npz."""
    D, sel = z["diffs"], z["sel_idx"]
    return np.array([D[p, sel[p]].mean() for p in range(D.shape[0])]), sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="+", required=True)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    report = {"maxt": [], "stability": None}

    profiles = []
    for path in a.records:
        z = np.load(path, allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        D, feas, gam = z["diffs"], z["feasible"], z["gammas"]
        pos = z["positions"]
        mask = feas & (gam[None, :] > 0)
        t_obs, mean, p_glob, p_adj, tmin = maxt(D, mask, a.boot, a.seed)
        cells = []
        for p in range(D.shape[0]):
            for g in range(D.shape[1]):
                if mask[p, g] and mean[p, g] < 0:
                    cells.append(dict(i=int(pos[p]),
                                      pair=f"{pos[p]+1}-{pos[p]+2}",
                                      gamma=float(gam[g]),
                                      dL=float(mean[p, g]),
                                      t=float(t_obs[p, g]),
                                      p_maxt=float(p_adj[p, g])))
        cells.sort(key=lambda c: c["t"])
        report["maxt"].append(dict(record=path,
                                   pool=meta.get("pool"),
                                   n_cells_tested=int(mask.sum()),
                                   t_min=float(tmin), p_global=p_glob,
                                   n_cells_p_maxt_lt_05=sum(
                                       c["p_maxt"] < 0.05 for c in cells),
                                   cells=cells[:40]))
        print(f"{path}: cells={int(mask.sum())} t_min={tmin:.3f} "
              f"global p={p_glob:.4f}; cells with max-t p<0.05: "
              f"{sum(c['p_maxt'] < 0.05 for c in cells)}")
        prof, sel = selected_profile(z)
        profiles.append((path, np.asarray(z["positions"]), prof,
                         np.asarray(z["gammas"])[sel] > 0))

    if len(profiles) > 1:
        try:
            from scipy.stats import spearmanr
        except Exception:
            spearmanr = None
        stab = []
        for i in range(len(profiles)):
            for j in range(i + 1, len(profiles)):
                _, pi, xi, gi = profiles[i]
                _, pj, xj, gj = profiles[j]
                common = np.intersect1d(pi, pj)
                ii = np.searchsorted(pi, common)
                jj = np.searchsorted(pj, common)
                rho = (float(spearmanr(xi[ii], xj[jj]).statistic)
                       if spearmanr else None)
                agree = float((gi[ii] == gj[jj]).mean())
                stab.append(dict(a=profiles[i][0], b=profiles[j][0],
                                 spearman_rho=rho,
                                 positive_gate_agreement=agree))
                print(f"stability {i}-{j}: spearman={rho} "
                      f"gate agreement={agree:.2f}")
        report["stability"] = stab

    json.dump(report, open(os.path.join(a.out, "maxt_report.json"), "w"),
              indent=1)
    print("->", os.path.join(a.out, "maxt_report.json"))


if __name__ == "__main__":
    main()
