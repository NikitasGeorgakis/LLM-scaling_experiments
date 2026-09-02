#!/usr/bin/env python3
"""Primary pre-registered analysis for preregistration_trajectory.md:
Kendall's tau between the small-gamma slope ghat_i and log(training tokens)
across Pythia-410m checkpoints, one-sided H1: tau > 0."""
import argparse
import glob
import os
import re

import numpy as np
from scipy.stats import kendalltau

STEPS = [512, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 143000]


def top_position_and_slope(npz_path, small_gamma_max=0.01):
    d = np.load(npz_path, allow_pickle=True)
    diffs = d["diffs"]
    gammas = d["gammas"]
    positions = d["positions"]

    g1_idx = int(np.argmin(np.abs(gammas - 1.0)))
    mean_at_g1 = diffs[:, g1_idx, :].mean(axis=1)
    top_pos_idx = int(np.argmin(mean_at_g1))
    top_position = int(positions[top_pos_idx])

    small_idx = [i for i, g in enumerate(gammas) if 0 < g <= small_gamma_max]
    g_vals = np.array([gammas[i] for i in small_idx])
    dl_vals = np.array([diffs[top_pos_idx, i, :].mean() for i in small_idx])
    ghat = float(np.sum(g_vals * dl_vals) / np.sum(g_vals ** 2))

    return top_position, ghat, mean_at_g1[top_pos_idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-glob", default="runs/prereg410m_step*/screen.npz")
    ap.add_argument("--small-gamma-max", type=float, default=0.01)
    args = ap.parse_args()

    rows = []
    for path in sorted(glob.glob(args.runs_glob)):
        m = re.search(r"step(\d+)", path)
        if not m:
            continue
        step = int(m.group(1))
        top_pos, ghat, dl_at_g1 = top_position_and_slope(path, args.small_gamma_max)
        rows.append((step, top_pos, ghat, dl_at_g1))

    rows.sort(key=lambda r: r[0])
    missing = set(STEPS) - {r[0] for r in rows}
    if missing:
        print(f"WARNING: missing checkpoints {sorted(missing)} -- found {len(rows)}/9.")

    print(f"{'step':>8} {'top_pos':>8} {'ghat (small-gamma slope)':>26} {'dL@gamma=1':>14}")
    for step, top_pos, ghat, dl1 in rows:
        print(f"{step:>8} {top_pos:>8} {ghat:>26.6e} {dl1:>14.4e}")

    steps = np.array([r[0] for r in rows], dtype=float)
    ghats = np.array([r[2] for r in rows])

    tau, p_two_sided = kendalltau(steps, ghats)
    p_one_sided = p_two_sided / 2 if tau > 0 else 1 - p_two_sided / 2

    print(f"\nPRIMARY ANALYSIS (pre-registered, the ONLY hypothesis test):")
    print(f"  Kendall's tau(step order, ghat_i) = {tau:+.4f}")
    print(f"  H1: tau > 0 (slope trends toward zero/positive with more training)")
    print(f"  one-sided p = {p_one_sided:.4f}")
    print(f"  (tau is rank-based, so ranking by raw step number is exactly")
    print(f"   equivalent to ranking by log(training_tokens) for Pythia's fixed")
    print(f"   batch/sequence length -- no token-count conversion needed)")

    early = [r for r in rows if r[0] <= 4000]
    if early:
        most_neg = min(early, key=lambda r: r[2])
        print(f"\nSECONDARY CONFIRMATION CANDIDATE (most negative ghat among "
              f"early checkpoints, step<=4000):")
        print(f"  step={most_neg[0]}, position={most_neg[1]}, ghat={most_neg[2]:.6e}")
        print(f"  -> run confirmation ONCE on this if desired, per the")
        print(f"     pre-registered secondary-confirmation clause.")


if __name__ == "__main__":
    main()
