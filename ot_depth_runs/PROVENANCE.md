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
Recorded in `pools/manifest.json`. Used for the E7 distribution-shift check
(reformulated from the original locked-candidate design; see ERRATA.md):
a full 35-position barycenter screen on gpt2-large, `runs/e7_openwebtext`.
0/35 positive gates.

## 5. G_wikitext

Built by the standalone `ot_depth_runs/make_wikitext_pool.py`, not
`make_pools.py`, for two reasons specific to this source:

1. HuggingFace's `wikitext` dataset requires an explicit config name
   (`wikitext-103-raw-v1`) that `make_pools.py`'s `hf:<dataset>[:split[:field]]`
   scheme has no slot for.
2. WikiText-103-raw is not document-per-row the way Pile/OpenWebText are --
   it streams as raw lines (mostly single short paragraphs, with blank-line
   gaps and `" = Article Title = "` headers). Feeding it through
   `make_pools.py`'s per-row logic as-is would treat each short paragraph as
   its own "document"; most would fall under `--min-chars` and be dropped,
   producing a sparse, structurally different pool from the others.

`make_wikitext_pool.py` instead accumulates consecutive non-blank lines into
pseudo-documents until reaching `--target-chars` (default 5000, chosen to
match D_sel/E_sel's ~5400-char mean document length), so the resulting pool
is structurally comparable to the other pools, not just same-format JSONL.

- Source: `wikitext-103-raw-v1`, **train** split (the **test** split was
  tried first and is too small: only 239/3000 docs came out of it at
  `target_chars=5000` -- WikiText-103's test split is ~245K tokens total,
  sized for perplexity benchmarking, not for a 3000-doc pool).
- `G_wikitext.jsonl`: 3000 docs, `target_chars=5000`, `min_chars=200`,
  SHA-256 `ad7de3da82c1eb0b97635cb74114840b7c1f7b0c6bd13dae9703aa344fcc7ec6`.
- Not recorded in `pools/manifest.json` (that file is written only by
  `make_pools.py`); provenance is this section plus the run script's own
  printed SHA-256 at build time.
- Used for the WikiText-103 arm of the E7 distribution-shift check: a full
  35-position barycenter screen on gpt2-large, `runs/e7_wikitext`. 0/35
  positive gates.

Both `pseudo_documents()` runs (building `G_wikitext.jsonl`, and separately
the OpenWebText build for `F_openwebtext.jsonl`) triggered a `Fatal Python
error: PyGILState_Release` / `Aborted (core dumped)` at interpreter
shutdown. Both times this was confirmed harmless: the pool file and its
printed SHA-256 were already complete and written to disk before the crash,
and `sha256sum` on the resulting file matched the printed hash exactly. This
is a known-benign shutdown-time issue in the `datasets`/`pyarrow` streaming
stack, not a data-integrity problem.

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
| F_openwebtext | 3000 | OpenWebText | Yes | E7 (0/35) |
| G_wikitext | 3000 | WikiText-103-raw-v1, train, pseudo-docs | No (built by `make_wikitext_pool.py`) | E7 (0/35) |

**When citing a hash for pool_A or pool_B, always state which version** (see
each result JSON's own `"pool_sha256"` field, which is authoritative for
that specific run).
