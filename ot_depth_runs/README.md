# Training-free depth-insertion experiment runners (E1--E7)

Self-contained scripts implementing the corrected selection/confirmation
protocol of the paper. Conventions match the recorded runs
(`full_scale_*.json`): insertion index `i` is the 0-indexed left block,
pair `(F_{i+1}, F_{i+2})`; gates on `SMALL_GRID = {0, 1e-4, 3e-4, 1e-3,
3e-3, 1e-2, 3e-2, 0.1}`; tolerances `D_KL, D_rep <= 0.05`; 25 selection /
40 confirmation batches of 8x1024 tokens; bootstrap 2,000 / 10,000.

## Requirements

```
pip install torch transformers scipy numpy
pip install datasets        # only for hf: pool sources
pip install "lm-eval>=0.4"  # only for E6
```

Everything runs in **float32** by default -- do not switch to bf16: the paired
diffs of interest are 1e-5..1e-3 nats. Tested end-to-end on CPU with tiny
models under torch 2.13 / transformers 5.16 (`python test_smoke.py`); run the
smoke test once in your H100 environment before anything else.

## Step 0 -- carve the pristine pools NOW (before any experiment runs)

Use the SAME held-out source file/stream that produced your original selection
(A) and confirmation (B) pools, with `--skip-docs` set beyond everything A and
B consumed. The manifest records source offsets and sha256 per pool, which is
the disjointness proof for the paper (E12).

```
python make_pools.py --source file:pile_holdout.jsonl --skip-docs <PAST_A_B> \
    --pool C_final=3000 --pool D_sel=1500 --pool E_sel=1500 \
    --pool G_pile=3000 --out pools/
python make_pools.py --source hf:Skylion007/openwebtext:train \
    --pool F_openwebtext=3000 --out pools/
```

`C_final` is the E1 third pool: it must be touched exactly once, by E1.
For the Pythia runs (E2/E4) the pools must be Pile **hold-out** text (as your
originals were), not Pile train, since Pythia pretrained on Pile train.

## Step 1 -- calibration (required before trusting the barycenter rebuild)

The deterministic constructions (copy/naive/hard-OT) are exactly reproducible.
The free-support barycenter here matches the documented hyperparameters
(tau=1/2, eta=0.05, 25 rounds, 80 log-domain Sinkhorn iters, index-midpoint
init, mean-normalized squared-Euclidean cost) but is a reimplementation.
Verify it against your recorded GPT-2 numbers on your ORIGINAL pool A:

```
python run_screen.py --model-key gpt2-large --construction barycenter \
    --pool pools/A_sel.jsonl --positions 11 --refine --refine-at 11 \
    --out runs/calibration_gpt2
```

Expected: gamma*=0.1 with dL_sel close to -3.0366e-3 at (F12,F13), refined
gamma about 0.3 with dL_sel close to -6.17e-3. If it disagrees materially,
the knobs that differ from your original implementation are `--cost-norm`,
the descriptor layout in `otdepth.mlp_descriptors`, and any unit
subsampling (the recorded matching *diagnostic* used a 4,096-unit cap;
`otdepth.matching_diagnostic` reproduces that). If you still have the
original barycenter block weights, prefer importing them for the locked
candidates (below) -- then calibration only needs to be approximate.

## Execution order and commands

**E3 -- method-specific position searches (run BEFORE E1):**
```
python run_screen.py --model-key gpt2-large --construction copy_next \
    --pool pools/A_sel.jsonl --refine --out runs/e3_copy_next
python run_screen.py --model-key gpt2-large --construction hard_ot \
    --pool pools/A_sel.jsonl --refine --out runs/e3_hard_ot
```
No `--confirm-pool` here: their single evaluation happens inside E1.

**Locked paper candidates at (F12,F13):**
```
python build_locked_gpt2.py --out runs/locked_gpt2 \
    [--import barycenter=path/to/original_bary_block.pt]
```

**E1 -- single pristine third-pool evaluation (one shot, all candidates):**
```
python run_final_pool.py --model-key gpt2-large --pool pools/C_final.jsonl \
    --n-batches 40 --candidates "runs/locked_gpt2/*.pt" "runs/e3_*/locked/*.pt" \
    --pairwise copy_next,hard_ot,barycenter,naive --out runs/e1_final
```
Decision rule (pre-registered): 95% paired bootstrap interval (10,000
resamples) below zero + stability; Holm-adjusted one-sided p reported across
candidates; direct method-to-method paired intervals included. Duplicate
constructions (paper-locked vs E3-locked) are auto-labelled `name@pair`.

**E2 -- corrected all-position screens on Pythia (+ refine + confirm):**
```
for M in pythia-410m pythia-1b pythia-1.4b pythia-2.8b; do
  python run_screen.py --model-key $M --construction barycenter \
      --pool pools/A_sel.jsonl --refine --confirm-pool pools/B_conf.jsonl \
      --out runs/e2_$M
done
```
(Reuse your original A/B pools for comparability of base losses, or fresh
Pile-holdout pools -- either is valid if pre-specified.)

**E4 -- corrected-protocol trajectory (nine Pythia-1.4B revisions):**
```
for S in step512 step1000 step2000 step4000 step8000 step16000 step32000 \
         step64000 step143000; do
  python run_screen.py --model-key pythia-1.4b --revision $S \
      --construction barycenter --pool pools/A_sel.jsonl --refine \
      --confirm-pool pools/B_conf.jsonl --out runs/e4_$S
done
```

**E5 -- multiplicity + profile stability (GPT-2 barycenter):**
```
python run_screen.py --model-key gpt2-large --construction barycenter \
    --pool pools/D_sel.jsonl --out runs/e5_poolD
python run_screen.py --model-key gpt2-large --construction barycenter \
    --pool pools/E_sel.jsonl --out runs/e5_poolE
python run_maxt.py --records runs/calibration_gpt2/screen.npz \
    runs/e5_poolD/screen.npz runs/e5_poolE/screen.npz --out runs/e5_maxt
```
The calibration run doubles as the pool-A record (it must then be run with
`--positions all`, i.e. drop `--positions 11` and rerun once in full).
`run_maxt.py` reports the joint (position x gate) max-t global p, single-step
adjusted p per cell, and cross-pool Spearman/sign agreement of the profiles.

**E6 -- downstream evaluation of the locked candidates:**
```
python run_downstream.py --model-key gpt2-large \
    --candidates "runs/locked_gpt2/*.pt" --out runs/e6_downstream
```

**E7 -- distribution-shift test (same evaluator, shifted pools):**
```
python run_final_pool.py --model-key gpt2-large --pool pools/F_openwebtext.jsonl \
    --n-batches 40 --candidates "runs/locked_gpt2/*.pt" --out runs/e7_owt
python run_final_pool.py --model-key gpt2-large --pool pools/G_pile.jsonl \
    --n-batches 40 --candidates "runs/locked_gpt2/*.pt" --out runs/e7_pile
```
Prediction under the shift hypothesis: the gains shrink on OpenWebText
(near-pretraining) relative to Pile.

## Outputs

Each screen writes `screen.csv` (per-position selected gate + CI),
`screen.npz` (per-batch diffs for every position x gamma -- the E5 input),
`summary.json` (schema close to `full_scale_*.json`, incl. base per-batch
losses, pool sha256, wallclock), `locked/<construction>.pt`, and
`confirm.json`. `run_final_pool` writes `final.json`/`final.csv` including
per-batch losses per candidate. Everything needed for E12 (hashes, seeds,
grids, wallclocks) is recorded.

## Practical notes

* Hard-OT on Mistral solves a 14,336x14,336 assignment per position
  (scipy LAP): expect minutes per position; the barycenter Sinkhorn at that
  width is similarly the dominant cost (your original corrected Mistral
  screen needed two wall-time jobs).
* Rough single-H100 budgets: E1 minutes; E3 ~2-3 h; E2 ~4-6 h total;
  E4 ~8-10 h; E5 ~2 h extra pools; E6/E7 ~1-2 h each.
* `--grid full` reproduces the legacy 11-point grid (0..1.0) if you want
  records directly comparable to the old JSONs.
* `--kl-probe-batches` controls the D_KL/D_rep probe (default: first batch).
* `run_screen.py --positions 3,4,11` restricts to a subset for quick tests;
  `test_smoke.py` validates the whole stack on tiny models in <1 min.
