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

**The finding across every architecture, position, checkpoint, and pool
tested is a consistent, materiality-gated null: gamma\*=0.** This is treated
as the primary result, not a negative outcome to be explained away -- the
two-stage protocol and certified safe-fallback guarantee are validated
extensively by this consistency. See the paper's Results and Limitations
sections for the full discussion, including the max-t multiplicity analysis
in `ot_depth_runs/` (E5) that illustrates why materiality gating, not
p-value alone, is essential to the protocol.

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

*[Pre-registration documents (trajectory, timestamp, checklist) to be added
here -- see the paper's pre-registration statement for the committed
protocol, materiality threshold (delta=1e-3 nats), and tie-break rule.]*

## Status

Working repository accompanying an in-preparation manuscript (target
venues: TMLR, ACL, EMNLP, NeurIPS). Contact: Nikitas Georgakis, EPFL.
