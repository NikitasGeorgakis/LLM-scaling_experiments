# Pool provenance

This file documents how every pool under `ot_depth_runs/pools/` was built,
in what order, and -- importantly -- that **`pool_A.jsonl` and
`pool_B.jsonl` exist in two versions** with different SHA-256 hashes across
the experiments in this repository. This is not an error; it is documented
here so a reviewer comparing hashes across run outputs is not misled.

## 1. Original selection/confirmation pools (`pool_A.jsonl`, `pool_B.jsonl`)

Built by `ot_depth_runs/export_pools_jsonl.py`, which decodes the already-
cached, tokenized gpt2-large pool from the `otbli` pipeline
(`otbli/results/pools_gpt2-large_seed1234.pt`) back to text via
`batch_decode`, and writes it in the `{"text": ...}` JSONL format
`otdepth.pack_pool()` expects. This choice -- deriving `pool_A`/`pool_B`
from text rather than re-streaming fresh documents -- was made so their
content matches what the original gpt2-large selection/confirmation runs
actually saw, up to detokenize/re-tokenize rounding.

- `pool_A.jsonl` (selection): 200 docs, SHA-256 `3f0e3eb913846bce...`
- `pool_B.jsonl` (confirmation): 320 docs, SHA-256 `d597032e11588f99...`
- Documents consumed from the source stream: 450 (with a 2x safety margin,
  `SKIP_DOCS=900` was fixed for all pools built after this point, guaranteeing
  disjointness -- see section 2).

**Used by:** the calibration run at i=11, and E3 (copy_next, hard_ot full
35-position screens).

## 2. Padding for cross-tokenizer robustness

When E2 (Pythia 410m/1b/1.4b/2.8b) was first attempted against the pools
above, `run_screen.py` failed: `pool /.../pool_A.jsonl: 182967 tokens <
204800; add documents to the pool`. Pythia's tokenizer (GPT-NeoX BPE)
produces fewer tokens than GPT-2's tokenizer for the same text (the pool
was originally sized to exactly fill gpt2-large's token requirement, with
no margin for a different vocabulary).

Fixed via `ot_depth_runs/pad_pools.py`, which appends additional documents
streamed fresh from the same Pile source, starting **after** the 450 docs
already consumed by the original pool_A/pool_B and staying **strictly
below** `SKIP_DOCS=900` (so C/D/E/G, built from stream position 900 onward,
remain provably disjoint from A and B):

- `pool_A.jsonl` extended: +150 docs from stream positions [450, 600) ->
  **350 docs total, new SHA-256 `f252518f92f819ba...`**
- `pool_B.jsonl` extended: +150 docs from stream positions [600, 750) ->
  **470 docs total, new SHA-256 `6147014e080668b647d3af0ae02067c409723a0cf0410a87e0b07a78215e37f6`**
- Next free (unused) stream position after padding: 750 (still 150 below
  the 900 boundary).

Both extensions were verified duplicate-free and length-distribution-
consistent with the pre-existing D_sel/E_sel pools (mean doc length within
~3% of each other) before being accepted.

**Used by:** E2 (all 4 Pythia sizes), E4 (Pythia-1.4b, all 9 checkpoints),
and the E5 `poolA` re-screen (`runs/e5_poolA`).

## 3. C_final / D_sel / E_sel / G_pile

Built by `ot_depth_runs/make_pools.py --source "$PILE_SOURCE" --skip-docs
"$SKIP_DOCS"` (SKIP_DOCS=900), fresh Pile documents disjoint from
pool_A/pool_B by construction (see section 2). Provenance (doc counts,
stream index ranges, SHA-256) for each is recorded automatically in
`pools/manifest.json`.

- `C_final.jsonl`: 3000 docs -- reserved for E1 pristine third-pool
  evaluation (not yet used against any locked candidate, since none of the
  screened constructions cleared the selection objective -- see ERRATA.md).
- `D_sel.jsonl`, `E_sel.jsonl`: 1500 docs each -- used in E5 alongside the
  padded pool_A for multiplicity-honest stability testing (`run_maxt.py`).
- `G_pile.jsonl`: 3000 docs -- reserved, not yet consumed by a run in this
  repository as of the last commit.

## 4. F_openwebtext

Built by `ot_depth_runs/make_pools.py --source hf:Skylion007/openwebtext:train`
(no skip needed -- disjoint by dataset, not by stream position). 3000 docs.
Recorded in `pools/manifest.json`. Reserved for the E7 distribution-shift
comparison; not yet consumed, since E7 as designed requires a locked
candidate and none exists (see ERRATA.md).

## Summary table

| Pool | Docs | Source | In `manifest.json`? | Used by |
|---|---|---|---|---|
| pool_A (orig.) | 200 | decoded from otbli `.pt` cache | No (built by `export_pools_jsonl.py`) | calibration, E3 |
| pool_A (padded) | 350 | orig. + Pile [450,600) | No | E2, E4, E5(poolA) |
| pool_B (orig.) | 320 | decoded from otbli `.pt` cache | No | calibration, E3 |
| pool_B (padded) | 470 | orig. + Pile [600,750) | No | E2, E4 |
| C_final | 3000 | Pile, skip=900 | Yes | reserved (E1) |
| D_sel | 1500 | Pile, skip=900 | Yes | E5 |
| E_sel | 1500 | Pile, skip=900 | Yes | E5 |
| G_pile | 3000 | Pile, skip=900 | Yes | reserved |
| F_openwebtext | 3000 | OpenWebText | Yes | reserved (E7) |

**When citing a hash for pool_A or pool_B, always state which version** (see
each result JSON's own `"pool_sha256"` field, which is authoritative for
that specific run).
