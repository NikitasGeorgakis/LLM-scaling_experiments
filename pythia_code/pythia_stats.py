"""Paired statistical evaluation machinery (shared with the toy experiments).

Design:
- The validation text stream is split in HALF: SELECTION pool contexts come
  only from the first half, CONFIRMATION pool contexts only from the second
  half (disjoint text). gamma/position are chosen on SELECTION; the single
  chosen candidate is then tested once on CONFIRMATION.
- All comparisons are PAIRED: candidate and base are evaluated on the exact
  same contexts; statistics are computed on per-batch loss differences
  d_b = L_cand,b - L_base,b, whose SE is far smaller than the unpaired SE.
- Decision rule (pre-registered, printed before the confirmation result is
  computed): substantive improvement iff, on the CONFIRMATION set,
     mean(d) < -delta_material   AND   bootstrap 95% CI upper bound < 0,
  with delta_material = 1e-3 nats fixed a priori (about 0.055% relative for
  a base loss of ~1.8; two orders of magnitude above the 5e-6 float noise
  observed earlier, well below the ~1e-2 harm of a full-strength layer).
"""
import math
import numpy as np

DELTA_MATERIAL = 1e-3  # nats, pre-registered


def paired_stats(base_arr, cand_arr, n_boot=10000, seed=2026):
    d = cand_arr - base_arr           # negative = improvement
    n = len(d)
    mean = d.mean()
    sd = d.std(ddof=1)
    se = sd / math.sqrt(n)
    t = mean / se if se > 0 else 0.0
    # one-sided p for H1: mean < 0 (normal approx)
    p = 0.5 * (1.0 + math.erf(t / math.sqrt(2)))
    rng = np.random.default_rng(seed)
    boots = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
    return dict(n=n, mean=mean, sd=sd, se=se, t=t, p_one_sided=p,
                ci95=(float(ci_lo), float(ci_hi)))


def print_rule():
    print("PRE-REGISTERED DECISION RULE (fixed before confirmation is computed):")
    print(f"  substantive improvement  <=>  on CONFIRMATION set:")
    print(f"    (i)  mean paired diff < -{DELTA_MATERIAL} nats  (materiality)")
    print(f"    (ii) bootstrap 95% CI upper bound < 0            (significance)")


def verdict(stats):
    material = stats["mean"] < -DELTA_MATERIAL
    signif = stats["ci95"][1] < 0
    return material and signif, material, signif
