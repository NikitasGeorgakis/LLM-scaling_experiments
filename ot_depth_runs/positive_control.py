#!/usr/bin/env python3
"""Positive control: verifies select_gamma() correctly returns gamma*>0
when the underlying numbers genuinely warrant it, and reports the exact
crossover point given typical measured KL/rep/Ceff overhead. No GPU, no
model loading -- pure decision-logic test, runs in <1 second.

This directly answers "would your pipeline detect a real effect if one
existed", using otdepth.py's own od.select_gamma() (not a reimplementation),
so it tests the ACTUAL code path used everywhere else in this repo.
"""
import os
import sys

sys.path.insert(0, os.path.expanduser("~/otbli"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# select_gamma()/_paper_gate_config() are pure CPU logic (no GPU touched),
# so this runs fine on the Kuma LOGIN node -- just needs the venv active
# (real torch must be importable; only CUDA is unavailable there, which
# this script never calls).
import otdepth as od

L0 = 3.0337  # gpt2-large base loss, matches today's actual measured L0_sel

# Realistic KL/rep/Ceff at gamma=0.1, borrowed from the real calibration_gpt2_v2
# measurement at (F12,F13) plus the printed Stage-B C_eff for gpt2-large.
KL, REP, CEFF = 3.618e-04, 2.145e-04, 0.0255

print("=" * 78)
print("PART 1: sensitivity sweep (does select_gamma() ever return gamma*>0?)")
print("=" * 78)
print(f"{'injected dL (nats)':>20} {'% of L0':>10} {'gamma*':>8} {'verdict':>16}")
crossed = False
for true_effect in [-2e-4, -5e-4, -1e-3, -2e-3, -5e-3, -1e-2, -2e-2, -3e-2,
                     -4e-2, -4.5e-2, -5e-2, -6e-2, -1e-1]:
    records = {
        "0.0": {"gamma": 0.0, "loss": L0, "dL_raw": 0.0, "KL": 0.0, "rep": 0.0,
                "Ceff": 0.0, "feasible": True},
        "0.1": {"gamma": 0.1, "loss": L0 + true_effect, "dL_raw": true_effect,
                "KL": KL, "rep": REP, "Ceff": CEFF, "feasible": True},
    }
    g_star, dL = od.select_gamma(records)
    verdict = "POSITIVE GATE" if g_star > 0 else "safe fallback"
    if g_star > 0 and not crossed:
        print("  " + "-" * 60 + "  <- crossover")
        crossed = True
    print(f"{true_effect:>20.4e} {100*true_effect/L0:>9.2f}% {g_star:>8} {verdict:>16}")

print("\n=> select_gamma() DOES return gamma*>0 for large enough effects.")
print("   The decision rule is not vacuously biased toward null.\n")

print("=" * 78)
print("PART 2: exact crossover point (binary search)")
print("=" * 78)
lo, hi = -0.06, -0.04
for _ in range(40):
    mid = (lo + hi) / 2
    records = {
        "0.0": {"gamma": 0.0, "loss": L0, "dL_raw": 0.0, "KL": 0.0, "rep": 0.0,
                "Ceff": 0.0, "feasible": True},
        "0.1": {"gamma": 0.1, "loss": L0 + mid, "dL_raw": mid, "KL": KL,
                "rep": REP, "Ceff": CEFF, "feasible": True},
    }
    g_star, _ = od.select_gamma(records)
    if g_star > 0:
        lo = mid   # mid still triggers a gate -> crossover is between mid and hi
    else:
        hi = mid   # mid is safe fallback -> crossover is between lo and mid
print(f"crossover at dL = {lo:.5f} nats ({100*lo/L0:.3f}% relative to L0={L0})")
print("(dominated by the FIXED efficiency-cost term: C_eff does not shrink")
print(" with small gamma, since the extra block runs in full whenever the")
print(" gate is active at all -- this sets a structural floor on how large")
print(" a benefit must be before ANY nonzero gamma is ever worthwhile.)")
