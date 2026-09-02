# OT barycentric layer insertion on Pythia — full-fidelity Kuma runbook

Complete, no-shortcuts implementation of the paper's method on real Pythia
checkpoints: the **entire** gate-selection machinery of Section 3.4 (not just
ΔL), the two-stage selection/confirmation protocol (eq. 3.48) with the
materiality margin (eq. 3.47), and the exact-layer-persistence needed for a
bit-reproducible confirmation.

## What "full fidelity" means here

| Paper element | Implementation |
|---|---|
| Grid Γ⁽⁰⁾, denser near 0 (eq. 3.29) | `pythia_gate.paper_grid()` — verbatim |
| ΔL normalized (eq. 3.30) | paired per-batch losses, normalized by |L(0)|+ε_L |
| D_KL^out (eq. 3.31) | exact mean KL(p₀‖p_γ) over all predictive positions; base logits computed once per batch and reused across the whole γ grid |
| D_rep (eq. 3.32) | probe hook measures D_rep(1) with the layer's true call args; D_rep(γ)=γ²·D_rep(1) exactly (§3.6 identity) |
| C_eff (eq. 3.33–3.35) | params + matmul-FLOPs proxy + **measured** latency + memory, weights ω; C_eff(0)=0 by bypass |
| Feasible set (eq. 3.36) | all four constraints; γ=0 always feasible |
| Objective J_γ (eq. 3.40) | all five λ-terms |
| Smallest minimizer (eq. 3.42) | tie-break to smallest γ |
| Grid refinement (eq. 3.43) | one pass, Q=8, around the coarse best |
| Two-stage protocol (eq. 3.48) | disjoint text halves; rule printed before result |
| Materiality (eq. 3.47) | δ = 10⁻³ nats, pre-registered |
| Exact confirmation | winning layer's state_dict saved at selection, loaded at confirmation — no rebuild drift |

All tolerances/weights (ε_KL=0.05, ε_rep=0.05, ε_eff=0.10; λ_L=1, λ_KL=0.1,
λ_rep=0.05, λ_eff=0.05, λ_γ=0.01; ω=¼ each) are pre-registered defaults,
printed at the start of every run, overridable via CLI flags.

## Verification status (be precise about this)

`test_integration.py` builds a REAL `GPTNeoXForCausalLM` (random weights, no
download) and exercises every code path. On transformers 5.15.0 / torch 2.13,
all six checks pass:

1. barycentric construction on a real GPTNeoXLayer (finite, correct shapes)
2. γ=0 identity: M⁺₀ == M **bit-exactly** (max diff 0.0)
3. O(γ) continuity
4. drift probe is pass-through and D_rep(γ)=γ²D_rep(1) holds
5. full gate selection (paper grid + refinement + smallest-min pick) end-to-end
   — and on random weights it correctly returns γ*=0 (safe fallback works)
6. save→load of the layer reproduces losses bit-exactly

**Not yet verified:** downloading actual Pythia weights and the two HF
datasets, and GPU execution. `huggingface.co` is network-blocked from where
this was written (confirmed: `curl -I https://huggingface.co` → `403
host_not_allowed`), so `pythia_prefetch.py` and the GPU runs are the first
real test of that path. Run prefetch, then the smoke test, before anything
that costs more than a few cents.

## Workflow on Kuma

```bash
# 0. upload this folder, then once:
bash setup_env.sh

# 1. integration test on the login node (free, ~1 min, no network needed)
source /scratch/$USER/pythia_venv/bin/activate
python test_integration.py          # must print ALL CHECKS PASSED

# 2. pre-fetch every model + dataset ONCE (try the login node first -- free)
export HF_HOME=/scratch/$USER/hf_cache
python pythia_prefetch.py
#   If this errors with a connection/timeout, the login node has no outbound
#   internet; fall back to a short interactive GPU session instead:
#     srun --partition=h100 --gpus=1 --time=00:30:00 --pty bash
#     source /scratch/$USER/pythia_venv/bin/activate
#     export HF_HOME=/scratch/$USER/hf_cache
#     python pythia_prefetch.py
#     exit
#   No HuggingFace account/token needed -- every repo used here is public.

# 3. smoke test on a GPU (~2 min, ~CHF 0.02; cache already warm from step 2)
srun --partition=h100 --gpus=1 --time=00:20:00 --pty bash
python pythia_select.py --smoke --out /tmp/smoke.json
exit

# 4. one real model
mkdir -p logs results
sbatch --job-name=p1b slurm_pythia.sh EleutherAI/pythia-1b

# 5. the scaling sweep (the scientifically decisive run)
bash submit_all.sh                  # 410m, 1b, 1.4b, 2.8b on the Pile
#   Safe to launch in parallel: the cache is already warm, so the four jobs
#   read the same files instead of racing to download them.

# 6. aggregate
python summarize.py
```

For headline numbers add `--strong` (sel=40/conf=60 batches, top-3 positions):
`sbatch --job-name=p1b slurm_pythia.sh EleutherAI/pythia-1b pile --strong`

## Files

| file | role |
|---|---|
| `pythia_wrapper.py` | model loading; gated insertion via signature-agnostic pre-hook; loss+logits; KL; drift probe; latency; param accounting; `from_model()` for tests |
| `pythia_barycentric.py` | log-domain Sinkhorn (numerically verified); free-support W₂ barycenter of MLP units; single-atom interpolation; baselines; Hungarian matching diagnostic (§3.6) |
| `pythia_gate.py` | the complete Section-3.4 machinery (see table above) |
| `pythia_data.py` | Pile (default, `NeelNanda/pile-10k`) or WikiText-103; disjoint halves |
| `pythia_stats.py` | paired stats, bootstrap CI, pre-registered rule — identical to the toy code |
| `pythia_select.py` | Stage A position screen + Stage B full gate selection; saves winning layer |
| `pythia_confirm.py` | loads exact layer; pre-registered test; descriptive KL/drift/latency; exploratory γ |
| `test_integration.py` | the six checks above |
| `pythia_prefetch.py` | one-time download of all models + datasets into the shared cache (no HF account needed; avoids the four sweep jobs racing to download the same files) |
| `summarize.py` | scaling-trend table across sizes |
| `setup_env.sh`, `slurm_pythia.sh`, `submit_all.sh` | environment + jobs |

## Cost & time (H100 @ CHF 0.5174/h)

Rough per-model estimates with defaults (25 sel / 40 conf batches, 8×1024):
410m ≈ 25 min, 1b ≈ 45 min, 1.4b ≈ 60 min, 2.8b ≈ 2–3 h (31 positions).
Whole sweep ≈ CHF 3–6. The 6 h `--time` ceiling caps any job at ~CHF 3.1.
For 2.8b you can halve Stage-A cost with `--positions 0,2,4,...` if needed.

## Interpreting the outcome

- `SUBSTANTIVE IMPROVEMENT` — mean < −10⁻³ nats AND CI95 < 0 out-of-sample:
  the strong-NeurIPS outcome.
- `STATISTICALLY DETECTABLE BUT NOT MATERIAL` — what the 12-layer toy gave.
- `NOT CONFIRMED` / `gamma*=0 at selection` — the safe fallback; the original
  model is recovered exactly.

The **trend across the four sizes** (summarize.py) is the decisive evidence
either way — a single size cannot separate a real effect from an artifact.

## Troubleshooting

- Hook errors about `hidden_states` → print `args`/`kwargs` inside the hook;
  it already handles positional and keyword passing on 5.15.0.
- OOM on 2.8b → `--batch 4` or `--block-size 512` (fp32 on 94 GB should fit).
- Sinkhorn NaN → raise `--eta` to 0.1 (log-domain solver makes this unlikely).
- Slow Hungarian on huge d_ff → it auto-subsamples to 4096 units.
