# OT-BLI: Optimal-Transport Barycentric Layer Insertion — Pythia experiments

Companion code for **"Scaling without re-training"** (N. Georgakis & V. Panaretos, EPFL, 2026).
This repository contains everything needed to reproduce the **full-scale evaluation on
pretrained Pythia checkpoints** (paper Section 6) and the **checkpoint-trajectory
follow-up** (Section 6.5), exactly as pre-registered.

The method: given two neighboring pretrained layers, atomize their parameters into
discrete measures, build a **free-support Wasserstein-2 barycenter** layer `G_bar`
(entropic OT, log-domain Sinkhorn), and insert it through a **scalar residual gate**
`G_γ(h) = (1−γ)h + γ G_bar(h)` selected by **forward-only grid search** — no gradient
step anywhere. `γ = 0` recovers the original model bit-exactly (safe fallback).

**Headline result reproduced here:** on fully-trained Pythia 410M–2.8B, the
pre-registered two-stage protocol returns **γ\* = 0 at every scale**; the small-γ
loss-only dips are noise-level below 1.4B and absent at 2.8B, and the strongest
candidate fails out-of-sample confirmation with a sign flip. The trajectory run
(`results/trajectory_pythia-1.4b_pile.csv`) extends the null across intermediate
checkpoints step512–step143000.

## Repository layout

```
otbli/
├── otbli/
│   ├── config.py       # ALL pre-registered constants (Section 6.1), printed before any run
│   ├── sinkhorn.py     # log-domain Sinkhorn with cost normalization (impl. note (i))
│   ├── barycenter.py   # free-support W2 barycenter, eqs. (3.15)–(3.18)
│   ├── atomize.py      # GPT-NeoX atomization A_b / reconstruction Q_b; naive & duplicate baselines
│   ├── insertion.py    # gated forward pre-hook, eq. (3.24); base weights never touched (note (ii))
│   ├── data.py         # held-out Pile pools: fixed-seed shuffle, 1024-token packing,
│   │                   #   25 selection + 40 disjoint confirmation batches of 8×1024 tokens
│   ├── metrics.py      # paired loss (3.30), output KL (3.32), C_eff (3.34), bootstrap, paired t
│   ├── protocol.py     # Stage A screen, Stage B gate machinery (3.29–3.47),
│   │                   #   loss-only secondary rule, confirmation rule (3.48)
│   ├── tasks.py        # lm-eval-harness backend: per-question scores, exact pairing
│   ├── task_protocol.py# the same two-stage protocol with task accuracy as the target
│   └── diagnostics.py  # exact recovery, drift identity γ²·D_rep(1) (note (iii)),
│                       #   exact-LAP matching diagnostic (3.49)
├── scripts/
│   ├── run_full_scale.py    # Section 6: pythia-410m / -1b / -1.4b / -2.8b (Tables 5–7 pipeline)
│   ├── run_confirmation.py  # the SINGLE pre-declared out-of-sample test, rule (3.48)
│   ├── run_trajectory.py    # Section 6.5 follow-up across HF revisions step512…step143000
│   ├── run_all.py           # unattended sweep over ALL models (resume-safe, one subprocess each)
│   ├── run_task_gate.py     # gated duplication judged on downstream task accuracy
│   └── build_pools.py       # prebuild pools on an internet-connected node (enables offline jobs)
├── slurm/
│   ├── setup_kuma.sh        # one-time frontend setup: venv on /scratch, deps, weight prefetch, pools
│   ├── kuma_run_all.sbatch  # SLURM job for the full sweep (partition h100, 1 GPU)
│   ├── kuma_task_gate.sbatch   # SLURM job for the task-accuracy experiment
│   └── kuma_trajectory.sbatch  # SLURM job for the Section-6.5 trajectory at the new scale
├── tests/smoke_test.py      # offline end-to-end test on a tiny random GPT-NeoX (no downloads)
└── results/
    └── trajectory_pythia-1.4b_pile.csv   # trajectory outcome to date (γ* = 0 at every step)
```

## Install

```bash
pip install -r requirements.txt
python tests/smoke_test.py        # offline integration test, runs on CPU in ~1 min
```

Tested with `torch 2.x` / `transformers 4.4x`. Evaluation is `float32` on a single
H100 (the complete four-model study is ≈2.5 GPU-hours; the pythia-1b selection stage
alone is ≈10 min).

## Usage

**Full-scale evaluation (Section 6, Tables 5–6 + diagnostics of 6.4):**

```bash
python scripts/run_full_scale.py \
  --models EleutherAI/pythia-410m EleutherAI/pythia-1b \
           EleutherAI/pythia-1.4b EleutherAI/pythia-2.8b \
  --device cuda --out results/
```

This prints the pre-registered constants, screens every valid insertion position at
γ=1 (Stage A), runs the complete gate machinery at the top-2 positions (Stage B:
grid (3.29), feasibility (3.37), objective (3.41), smallest-minimizer tie-break
(3.43), one refinement pass (3.44), materiality rule (3.47) with δ = 10⁻³ nats),
then runs the mechanism diagnostics (bit-exact recovery of `M`, the drift identity,
the exact-LAP matching diagnostic, and the baseline ordering *barycentric ≤ naive <
duplicate*). Pools are cached to `results/pools_<model>_seed1234.pt`; **the
confirmation pool is written but never read** by this script.

**The single confirmation (Section 6.3)** — decide the candidate from the selection
records first; the confirmation pool is touched exactly once:

```bash
python scripts/run_confirmation.py --model EleutherAI/pythia-1.4b \
  --position 1 --gamma 0.01 --pools results/pools_pythia-1.4b_seed1234.pt
```

(`--position` is 0-indexed: `--position 1` is the paper's pair (F2, F3).)

**Trajectory follow-up (Section 6.5):**

```bash
python scripts/run_trajectory.py --model EleutherAI/pythia-1.4b \
  --steps 512 1000 2000 4000 8000 16000 32000 64000 143000 \
  --device cuda --dataset-label pile
```

Writes the compact CSV (`step,gamma,delta_L,tau,model,dataset` — the schema of
`results/trajectory_pythia-1.4b_pile.csv`) **and** an `_extended.csv` that records,
per checkpoint, the most negative stability-feasible loss-only dip with its bootstrap
CI — the quantity the slack hypothesis actually predicts should decay with training.

## Running everything unattended

`scripts/run_all.py` sweeps the whole Pythia suite (70M → 6.9B by default) with one
subprocess per model — a failure or OOM on one model cannot kill the rest, GPU memory
is fully released between models, and the sweep is **resume-safe**: models whose
`results/full_scale_<model>.json` already exists are skipped, so a killed run is
simply resubmitted. It ends by writing `results/summary_all.csv`.

On your own machine (survives logout, but the machine must stay powered on —
shutting it down or letting it sleep kills the run):

```bash
mkdir -p logs
nohup python scripts/run_all.py --device cuda > logs/run_all.log 2>&1 &
tail -f logs/run_all.log
```

To be independent of your own computer entirely, run it on a cluster (below).

## Running on SCITAS Kuma (EPFL)

Kuma is EPFL's GPU cluster (login `kuma.hpc.epfl.ch`; partitions `h100` with
4× H100 94 GB per node and `l40s`; QOS `normal` ≤ 3 days / `long` ≤ 7; the partition
is mandatory and the default walltime is 5 minutes, so the provided sbatch files set
both). A single H100 94 GB holds pythia-6.9b in float32 with plenty of headroom
(and even 12b at 48 GB of weights).

```bash
ssh <gaspar>@kuma.hpc.epfl.ch
git clone https://github.com/<user>/otbli.git && cd otbli

# one-time, on the frontend (has internet): venv on /scratch, deps,
# model-weight prefetch, pool prebuild -> jobs then run fully offline
bash slurm/setup_kuma.sh
# optionally also prefetch the 9 trajectory checkpoints (~125 GB on scratch):
TRAJ_MODEL=EleutherAI/pythia-6.9b bash slurm/setup_kuma.sh

# submit; the job runs on the cluster, independent of your own computer
sbatch slurm/kuma_run_all.sbatch
sbatch slurm/kuma_trajectory.sbatch        # optional Section-6.5 run at 6.9B

squeue -u $USER                            # monitor
tail -f logs/otbli-all_<jobid>.out
```

Results land in `results/` inside the repo; copy them back with
`scp -r <gaspar>@kuma.hpc.epfl.ch:otbli/results ./results-kuma` (or commit them).
Note that `/scratch` is periodically purged — the venv and HF cache are cheap to
recreate with `setup_kuma.sh`; keep the repo and results in `$HOME`.

### Troubleshooting (field notes from real Kuma runs)

All of the following were hit in practice and their fixes are already baked into this
repo; they're listed so the symptoms are recognizable.

- **HF `429 Too Many Requests` during prefetch** — the cluster's shared egress IP hits
  HuggingFace's anonymous rate limit. Create a free Read token at
  https://huggingface.co/settings/tokens, `export HF_TOKEN=hf_...` in the same shell,
  re-run `setup_kuma.sh` (cached files are skipped). The prefetch also retries with
  backoff and fetches only the needed files, not onnx/tf/flax siblings.
- **`ValueError: Compression type zstd not supported` while building pools** — the
  Pile mirror ships `.jsonl.zst` shards; `zstandard` is now in `requirements.txt`
  (in an existing venv: `pip install zstandard`).
- **`RuntimeError: The NVIDIA driver on your system is too old`** — PyPI's default
  torch wheel targets a newer CUDA than Kuma's driver stack. `requirements.txt` now
  pins the cu124 wheel index (matching Kuma's CUDA 12.4). In an existing venv:
  `pip install torch --index-url https://download.pytorch.org/whl/cu124`.
- **`sbatch` rejects the job (`MaxMemPerCPU limit (5900.0)` / `more than the maximum
  allowed of 16 cores per GPU`)** — Kuma allocates RAM per core (5900 MB) and caps
  cores at 16 per GPU, so ≤ ~92 GB host RAM per single-GPU job. The sbatch files now
  request `--cpus-per-task=16 --mem=88G` (cpu.hours are not billed on GPU clusters,
  only gpu.hours, so extra cores cost nothing).

## Second architecture: which model tests the THEORY, and how

The theoretical core (Section 3.6) lives at the level of exchangeable MLP units, so a
second-architecture run must keep the unit-level OT construction — not bypass it.
Three complementary choices, all supported end to end:

1. **`openai-community/gpt2-large` (or `gpt2-xl`)** — *verbatim transfer.* Same simple
   2-layer GELU MLP and pre-LN structure as Pythia, so Propositions 3.1/3.2 and the
   GELU constants apply word for word; different tokenizer and training data test
   whether the Pythia null is family-specific. (GPT-2 is named in the paper's own
   Evaluation Plan.)
2. **`TinyLlama/TinyLlama_v1.1` → `mistralai/Mistral-7B-v0.1`** — *theory extension to
   SwiGLU.* A gated unit computes `down_r · SiLU(gate_r·h) · (up_r·h)`, so the layer is
   still a linear functional of a measure over coupled triples `(gate_r, up_r, down_r)
   ∈ R^{3d}` and the whole formalism extends — **but** `(up_r, down_r) → (λ·up_r,
   down_r/λ)` is an exact function-preserving gauge, precisely remark (i) of Section
   3.6. The atomization therefore gauge-fixes (`‖up_r‖ = 1`, scale folded into
   `down_r`) before transport; `OTConfig.gauge_fix = False` ablates this, and the
   matching diagnostic reports both, turning the remark into a measurable prediction.
3. *(optional stress test)* an OPT checkpoint — ReLU makes the `(u, v) → (λu, v/λ)`
   gauge exact for the simple MLP too, probing the same remark from the other side.

```bash
# theory-transfer control + SwiGLU extension (TinyLlama first: cheap sanity, ~minutes)
python scripts/run_full_scale.py --models openai-community/gpt2-large \
    TinyLlama/TinyLlama_v1.1 mistralai/Mistral-7B-v0.1 --device cuda --out results
```

Llama-2/3 checkpoints work identically (`meta-llama/Llama-2-7b-hf`) but are
license-gated on the Hub: run `huggingface-cli login` with an approved token first.
GPT-2, TinyLlama and Mistral-7B-v0.1 need no token. Adding further architectures is a
three-step edit documented in **[ARCHITECTURES.md](./ARCHITECTURES.md)**.

## Downstream-task target metric: gated duplication (`run_task_gate.py`)

The Section-6 runs target held-out loss and return gamma* = 0 everywhere. The one
published training-free depth edit with a *measurable gain* on converged LLMs points
the other way: duplicating a single early layer, scored on **tasks** rather than loss
(Eyal, Dershowitz & Bar, Findings of EMNLP 2025 — ~+8.8% mean relative gain over 10
BigBench tasks on Pythia 70M-1B, +6.8% on Llama-3.1-8B, zero training). In our own
loss-targeted runs the duplicate baseline is the *worst* arm (dL = +0.03..+0.08 nats),
so the two metrics are expected to disagree.

`scripts/run_task_gate.py` puts that disagreement inside the pre-registered machinery.
G_bar becomes a verbatim copy of an early block (1-3), the target metric becomes task
accuracy scored per question through lm-evaluation-harness, and everything else is
unchanged: the same grid (3.29), feasibility screen (3.37), objective (3.41) with the
loss term replaced by the normalized accuracy term, tie-break (3.43), refinement
(3.44), materiality (3.47) with delta = 1 accuracy point, and confirmation (3.48) on a
disjoint, stratified half of the questions — inequalities flipped, since accuracy is
better when larger. Pairing is exact: the same task, doc_id and few-shot context are
compared across gammas, and the run aborts if the question set moves.

**What is new relative to the published result.** Every grid point records the task
gain *and* the held-out loss cost, so the output is a trade-off curve rather than a
single edited model — the run locates whether some gamma buys the task gain at a
negligible loss cost, or whether gain and cost are inseparable. The statistics are
also stricter: per-question pairing with a bootstrap CI plus an exact McNemar test on
the discordant pairs, and a single pre-declared out-of-sample confirmation. On
simulated data at the published effect size the rule has ~2% false-positive rate under
a true null and ~99% power at +2.5 points with 1,000 questions.

```bash
pip install 'lm-eval>=0.4.2'
python scripts/run_task_gate.py --list-tasks bigbench | head -40   # verify task names
python scripts/run_task_gate.py --models EleutherAI/pythia-410m EleutherAI/pythia-1b \
    --device cuda --limit 200 --out results
```

Then, for the surviving candidate only, **once**:

```bash
python scripts/run_task_gate.py --models EleutherAI/pythia-1b --confirm \
    --position 0 --gamma 0.5 --device cuda --limit 200 --out results
```

On Kuma: `sbatch slurm/kuma_task_gate.sbatch`. Task names differ between harness
versions, so always check `--list-tasks` first; only likelihood-scored (multiple
choice) tasks are supported, since the gate is defined for teacher-forced evaluation.
`--kl-free` ablates the output-KL feasibility screen, which an edit that changes task
behaviour is expected to violate.

## Pre-registered constants (config.py)

| group | values |
|---|---|
| barycenter | τ = 1/2, η_b = 0.05 (mean-normalized cost), 25 alternating rounds × 80 Sinkhorn iters |
| gate grid | {0, 10⁻⁴, 3·10⁻⁴, 10⁻³, 3·10⁻³, 10⁻², 3·10⁻², 10⁻¹, 0.2, 0.5, 1} |
| tolerances | ε_KL = ε_rep = 0.05, ε_eff = 0.10 |
| objective | λ_L = 1, λ_KL = 0.1, λ_rep = λ_eff = 0.05, λ_γ = 0.01; ω_F = ω_T = ω_M = ω_P = 1/4 |
| materiality | δ = 10⁻³ nats (eq. 3.47); rule (3.48): d̄ < −δ **and** CI⁹⁵ᵘᵖᵖᵉʳ < 0 |
| pools | seed 1234, blocks of 1024 tokens; 25 selection + 40 confirmation batches of 8 blocks |
| bootstrap | 2,000 resamples (selection), 10,000 (confirmation) |

## Data note

The paper evaluates on held-out shards of **The Pile** tokenized with each model's
own tokenizer. `otbli/data.py` accepts either a HuggingFace dataset name (default:
the `monology/pile-uncopyrighted` mirror, since the original Pile hosting is
intermittent) or a local `.jsonl` path — pass your own held-out shards via
`--dataset /path/to/pile_heldout.jsonl` to match the paper's setup exactly.

## Implementation notes that matter at LLM scale (Section 6.1)

1. **Log-domain Sinkhorn with cost normalization** — the naive Gibbs kernel
   `exp(−C/η)` underflows to exact zero at LLM widths.
2. **Insertion as a forward pre-hook** before block *i+1* — base weights are never
   touched; at γ = 0 the hook is not even registered, so `M⁺₀ ≡ M` bit-exactly
   (verified by `diagnostics.exact_recovery_check` + weight fingerprints).
3. **Drift identity** — `D_rep(γ) = γ²·D_rep(1)` exactly; measured once per
   position and extended analytically (and checked numerically).

## Citation

```bibtex
@article{georgakis2026scaling,
  title  = {Scaling without re-training},
  author = {Georgakis, Nikitas and Panaretos, Victor},
  year   = {2026},
  note   = {EPFL, Institute of Mathematics}
}
```
