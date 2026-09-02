# Errata: gamma* selection rule in `ot_depth_runs`

**Status:** Fixed and re-verified. This file documents a bug found and
corrected on 2026-08-30/31, before any of its outputs were used in the
paper. It is kept in the repository for full transparency.

## Summary

`ot_depth_runs/otdepth.py`'s original `select_gamma()` chose the retained
gate gamma\* by **argmin of raw paired loss change alone**, subject to a
KL/representation-drift feasibility check. This is the paper's "loss-only
secondary candidate" heuristic (see `otbli/otbli/protocol.py`, where the
equivalent quantity is computed and explicitly kept separate from the
primary selection rule) -- not gamma\* under the pre-registered multi-term
objective *J* (eq. 3.41) with its certified safe-fallback guarantee (eq.
3.47). The bug had two parts:

1. **Wrong objective.** `select_gamma()` picked `argmin(dL_raw)` over
   feasible gammas, rather than `argmin(J)`, so it ignored the KL,
   representation-drift, and efficiency-cost terms and the `lambda_gamma *
   gamma^2` regularizer entirely when choosing *which* gamma to report.
2. **Missing terms.** `eval_candidate()` never computed the efficiency-cost
   term C_eff at all (no FLOPs/latency/memory/param accounting), so
   feasibility was checked against only 2 of the paper's 3 constraints
   (KL <= eps_KL and rep <= eps_rep, missing C_eff <= eps_eff). Separately,
   the materiality check (eq. 3.47, "reject back to gamma\*=0 unless the
   point clears delta in raw loss improvement") was never applied to the
   selected point at all.

## How it was found

A calibration cross-check for gpt2-large at (F12,F13) -- the position
targeted for the multi-architecture paper -- was run via `run_screen.py`
against `otbli`'s own independently-validated `run_full_scale.py --positions
11`, using otbli's native tokenized pool (no reconstruction). The
*measurements* (dL, KL, rep) agreed almost exactly between the two
pipelines at gamma=0.1 (otbli: dL=-3.037e-3 nats; `ot_depth_runs`/RUNBOOK
target: dL~=-3.04e-3 nats). But the *verdicts* diverged: `otbli` returned
gamma\*=0 (safe fallback fires) while `ot_depth_runs` returned gamma\*=0.1,
refining to gamma\*\*=0.3. Reading `otdepth.py`'s source confirmed the
selection rule was the loss-only proxy described above.

## What this affects

- The original hardcoded gammas in `build_locked_gpt2.py` --
  `{"copy_next": 0.5, "hard_ot": 0.5, "barycenter": 0.3, "naive": 0.3}` --
  were traced to this same pre-fix logic and were **not** derived from the
  corrected `select_gamma()`.
- A real, legitimately-run disjoint-pool confirmation exists on file
  (`otbli/results/gpt2_confirmation_f12_f13_g03/confirmation_gpt2-large_i11_g0p3.json`):
  d_mean=-1.5735e-3 nats, CI95=[-3.692e-3, -2.117e-6], p_one_sided=0.0542,
  `"confirmed": true`. This is a genuine, disjoint-pool-validated loss
  effect at (i=11, gamma=0.3) under rule (3.48) -- **but recomputing J from
  this confirmation file's own numbers gives J(0.3) = +7.61e-3, strictly
  greater than J(0)=0**, even before adding C_eff (which can only increase
  J further). Under the corrected selection objective, gamma=0.3 would
  never have been proposed as a candidate for confirmation in the first
  place. **Decision: this result is not carried forward as the paper's
  primary gpt2-large finding.** It remains on disk as a legitimate,
  documented loss-effect measurement, should it be useful as a secondary,
  clearly-labeled data point.

## The fix

`eval_candidate()` now computes C_eff (FLOPs/memory/param fraction via the
candidate block's own parameter count over the full model's, plus a real
wall-clock latency comparison), matching `omega = (1/4, 1/4, 1/4, 1/4)`.
`select_gamma()` now computes the full objective J using
`lambda_L=1, lambda_KL=0.1, lambda_rep=lambda_eff=0.05, lambda_gamma=0.01`,
**imported directly from `otbli.config.GateConfig`** (not a second,
independently-hardcoded copy) so the two packages cannot silently diverge
again, does smallest-gamma tie-break among J-minimizing feasible points (eq.
3.43), then applies the eq. (3.47) materiality check (delta=1e-3 nats)
before returning a nonzero gamma\*. `build_locked_gpt2.py` was rewritten to
determine gamma dynamically via `od.select_gamma()` rather than using
hardcoded values, and now explicitly reports and skips any construction
whose gamma\* is 0 rather than locking it.

Validated: the corrected `select_gamma()` was checked offline against the
gpt2-large (F12,F13) measurements above and reproduces gamma\*=0, matching
`otbli`'s verdict, robustly across a range of plausible `eps_L` values.

## Post-fix verification (2026-08-30/31)

Every experiment below was run (or re-run) after the fix:

| Experiment | Model(s) | Positions | Result |
|---|---|---|---|
| Calibration | gpt2-large | i=11 (F12,F13) | gamma\*=0, matches otbli exactly |
| E3 (copy_next) | gpt2-large | all 35 | 0/35 positive gates |
| E3 (hard_ot) | gpt2-large | all 35 | 0/35 positive gates |
| E1 (build_locked_gpt2, dynamic) | gpt2-large | i=11, 4 constructions | 0/4 locked |
| E2 | Pythia 410m/1b/1.4b/2.8b | all positions (23+15+23+31=92) | 0/92 positive gates |
| E4 | Pythia-1.4b, 9 checkpoints (step512-step143000) | all 23 per checkpoint | 0/9 checkpoints, 0/207 position-checkpoint cells |
| E5 | gpt2-large, 3 independent pools (A/D/E) | all 35 per pool | 0/105 positive gates; positive-gate agreement 1.0 across all pool pairs |

E5's max-t (multiplicity-adjusted) analysis, run on the pre-materiality-
filter raw grid, shows inconsistent nominal significance across the three
pools (poolA: 12/245 cells p_maxt<0.05; poolD: 0/245; poolE: 103/245) --
but every flagged cell's raw effect size is 10-100x smaller than the
delta=1e-3 nats materiality threshold (e.g. -1.9e-5 to -6.8e-5 nats). This
is consistent with small-sample statistical noise in the raw grid, not a
real effect, and is exactly the scenario the materiality gate (not p-value
alone) is designed to guard against. See the paper's Limitations section.

## Timeline

- Original (buggy) `ot_depth_runs` package received and deployed: 2026-08-30.
- Bug found via gpt2-large (F12,F13) cross-check: 2026-08-30.
- Fix implemented, verified offline, deployed to Kuma: 2026-08-30.
- Calibration re-verified against `otbli`: 2026-08-30.
- E1/E3 re-run under corrected rule: 2026-08-30.
- E2, E4, E5 run under corrected rule: 2026-08-30/31.
- E6, E7, pre-registered trajectory test run under corrected rule: 2026-08-31/09-01/09-02.
- Second, smaller gap found via a third, independent implementation
  (`pythia_gate.py`, a separate working directory not otherwise part of
  this repository): 2026-09-02.
- No results computed under the buggy rule were used in the paper.

## Addendum (2026-09-02): missing `dL_raw <= 0` feasibility term

`eval_candidate()`'s feasibility check was missing one of the four terms
implied by eq. (3.36)/(3.37): a candidate must not increase the raw loss
at all (`dL_raw <= 0.0`), independent of the KL/representation-drift/
efficiency-cost bounds. Found by comparing against a third, independent
implementation of the same protocol (`pythia_gate.py`, in a separate
working directory, `~/pythia_code/` on Kuma, not part of this repository)
during an unrelated provenance check -- that implementation's feasibility
check (`st["mean"] <= 0.0 and kl <= ... and drift <= ... and ce <= ...`)
includes this term explicitly; `otbli/otbli/task_protocol.py` (the
downstream-task variant) already had the correct accuracy-domain analogue
(`rec["acc"] >= A0`, eq. 3.37) from the start, so E6 is unaffected.

**This cannot change any verdict already reported in this repository.**
Adding a feasibility constraint only ever shrinks the feasible set (never
grows it), and gamma=0 -- which is always feasible with J=0 by construction
-- had already won argmin(J) in every single result in E1-E7 and the
pre-registered trajectory test (no candidate anywhere had J < 0 under the
existing, looser constraints). A stricter feasible set therefore keeps the
same winner in every case already computed. Fixed in `otdepth.py`
(`eval_candidate()`); no re-runs were required, but the fix is in place for
anything run after 2026-09-02.
