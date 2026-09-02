# OT-BLI: Training-free depth up-scaling via Optimal-Transport Barycentric Layer Insertion

Code and results supporting the paper's scaling experiments. This repo
contains two packages that share a single source of truth for the
pre-registered gate objective (`otbli.config.GateConfig`):

- **`otbli/`** -- the core OT-BLI method: builds a barycentric layer between
  adjacent frozen transformer blocks via a residual gate `G_{m,gamma}(h) =
  (1-gamma)h + gamma*G_m(h)`, and the primary two-stage
  screening/confirmation protocol (Sections 3-6 of the paper). Evaluated on
  Pythia (70M-1B), GPT-2 Large, TinyLlama, and Mistral-7B.
- **`ot_depth_runs/`** -- a second, independent screening pipeline used for
  the extended experiment set (E1-E7 in the paper's Section 11): additional
  candidate constructions (`copy_next`, `hard_ot`, `naive`, alongside
  `barycenter`), multi-pool stability testing, and multiplicity-honest
  (max-t) inference.

## Results summary

Every experiment below used the pre-registered multi-term objective *J*
(eq. 3.41), with its certified safe-fallback guarantee (eq. 3.47): the
retained gate defaults to gamma\*=0 (exact base-model recovery) unless a
candidate's loss improvement clears the delta=1e-3 nats materiality
threshold **after** accounting for KL divergence, representation drift, and
efficiency cost.

| Experiment | Model(s) | Scope | Result |
|---|---|---|---|
| Full-scale screen | GPT-2 Large, TinyLlama, Mistral-7B | all positions | gamma\*=0 (all three) |
| E1 | GPT-2 Large | (F12,F13), 4 constructions | 0/4 cleared selection |
| E2 | Pythia 410m/1b/1.4b/2.8b | all positions (92 total) | 0/92 positive gates |
| E3 | GPT-2 Large | all 35 positions, copy_next + hard_ot | 0/70 positive gates |
| E4 | Pythia-1.4b, 9 training checkpoints | all positions per checkpoint | 0/9 checkpoints |
| E5 | GPT-2 Large, 3 independent selection pools | all 35 positions per pool | 0/105 positive gates |
| E6 | Pythia 410m/1b (downstream accuracy) | 14 tasks (10 BigBench + 4 controls) | 0/2 gates retained |
| E7 (OpenWebText) | GPT-2 Large | all 35 positions | 0/35 positive gates |
| E7 (WikiText-103) | GPT-2 Large | all 35 positions | 0/35 positive gates |
| Pre-registered trajectory test | Pythia-410m, 9 checkpoints | Kendall's tau, small-gamma slope vs. training | tau=-0.667, one-sided p=0.9937 (H1 rejected) |

**The finding across every architecture, position, checkpoint, and pool
tested is a consistent, materiality-gated null: gamma\*=0.** This is treated
as the primary result, not a negative outcome to be explained away -- the
two-stage protocol and certified safe-fallback guarantee are validated
extensively by this consistency. See the paper's Results and Limitations
sections for the full discussion, including the max-t multiplicity analysis
in `ot_depth_runs/` (E5) that illustrates why materiality gating, not
p-value alone, is essential to the protocol. The pre-registered trajectory
test (see Pre-registration below) is a qualitatively stronger form of
evidence than the other rows: it commits to a single directional
hypothesis before running anything, and the hypothesis is cleanly rejected
rather than merely unconfirmed.

## A bug was found and fixed during this work

`ot_depth_runs/otdepth.py`'s original gate-selection rule did not implement
the pre-registered objective *J* -- it used a simpler loss-only heuristic
that the paper's own `otbli` protocol treats as an explicitly secondary,
non-primary signal. This was caught via a cross-check against `otbli`'s
independently-validated implementation, fixed, and every experiment above
was run (or re-run) under the corrected rule. Full details, including exact
numbers before and after the fix and what was and was not affected:
**[`ot_depth_runs/ERRATA.md`](ot_depth_runs/ERRATA.md)**.

## Pool construction and provenance

`ot_depth_runs/pools/pool_A.jsonl` and `pool_B.jsonl` exist in two versions
across the experiments in this repo (an original version and a version
padded for cross-tokenizer robustness) -- documented, with exact SHA-256
hashes and which experiments used which version, in
**[`ot_depth_runs/PROVENANCE.md`](ot_depth_runs/PROVENANCE.md)**.

## Reproducing

Both packages were run on SCITAS Kuma (EPFL), H100 partition. See:
- `otbli/requirements.txt`, `ot_depth_runs/requirements.txt` for the Python
  environment (PyTorch built against CUDA 12.4).
- `otbli/slurm/*.sbatch`, `ot_depth_runs/*.sbatch` for the exact SLURM job
  scripts used for every result above.
- `ot_depth_runs/RUNBOOK.md` for the full step-by-step sequence (pool
  construction, calibration, E1-E7), including the fix note at the top.
- `ot_depth_runs/pools/manifest.json` for pool construction provenance
  (document counts, source stream index ranges, SHA-256 hashes).

## Pre-registration

`ot_depth_runs/preregistration_trajectory.md` (locked 2026-08-16, before any
evaluation ran; SHA-256 and UTC timestamp in
`ot_depth_runs/preregistration_timestamp.txt`) commits to a single,
falsifiable trend test (paper Section 6.5): does the small-gamma slope of
the loss response decay toward zero/positive as Pythia-410m trains, per a
one-sided Kendall's tau test?

**Result: no.** `ot_depth_runs/prereg_trajectory_test_output.txt` (script:
`prereg_trajectory_test.py`) reports **tau = -0.667, one-sided p = 0.9937**
against the pre-registered H1 (tau > 0) -- a clean rejection in the
direction opposite the hypothesis, not an inconclusive result. The
pre-registered secondary confirmation (the single most negative slope among
early checkpoints, step=4000 at position 7) was run once as committed:
gamma\*=0, not retained (`runs/prereg410m_secondary_confirm`).

Both the primary screen (`runs/prereg410m_step*`) and the secondary
confirmation used this repository's corrected `ot_depth_runs` pipeline
(see ERRATA.md) to compute the pre-registered objective -- not the
`pythia_select.py`/`pythia_confirm.py` scripts named in `CHECKLIST.md`,
which come from an earlier, independent implementation of the same
protocol (not yet included in this repository). That earlier work ran
static selection+confirmation across all four Pythia sizes and a separate
1.4b trajectory sweep, both consistent with the null found throughout this
repository -- but neither is the specific 410m/Kendall's-tau trend test
this pre-registration commits to, which is the analysis above.

## Status

Working repository accompanying an in-preparation manuscript (target
venues: TMLR, ACL, EMNLP, NeurIPS). Contact: Nikitas Georgakis, EPFL.
