# Pre-registration: Small-γ Slope Trajectory Test (Pythia-410M)

Locked before any evaluation is run. See §6.5 of the paper ("A falsifiable follow-up").

## Objective
Test whether the small-γ slope ĝ_i (first-order signal, ΔL̂(γ) ≈ γ ĝ_i) decays
monotonically toward zero/positive as training progresses, per the slack hypothesis.

## Fixed design
- Model: EleutherAI/pythia-410m (pilot)
- Checkpoints (revisions): step512, step1000, step2000, step4000, step8000,
  step16000, step32000, step64000, step143000 (9 log-spaced points of 154 available)
- Insertion positions: screened as in Section 6 (Stage A -> top-k -> Stage B)
- Fixed constants (unchanged from Section 6):
  delta=1e-3, epsKL=epsrep=0.05, epseff=0.10
  lamL=1, lamKL=0.1, lamrep=lameff=0.05, lamgamma=0.01
  omegaF=omegaT=omegaM=omegaP=0.25
  tau=0.5, eta_b=0.05, 25 alternating rounds, 80 Sinkhorn iters/round

## Primary analysis (pre-declared, the ONLY hypothesis test)
For each checkpoint, extract ĝ_i = small-gamma slope of paired ΔLsel(gamma) vs
gamma (linear fit through origin, gamma <= 0.01, top screened position).
ONE trend test: Kendall's tau between ĝ_i and log(training tokens), one-sided
H1: tau > 0 (slope trends toward zero/positive with more training).

## Secondary confirmation (at most one)
Take the single most-negative ĝ_i among early checkpoints; run the unchanged
two-stage confirmation rule (eq. 3.48) once on its disjoint confirmation pool.
Report outcome regardless of result.
