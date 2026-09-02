# CHECKLIST — gated execution plan (do not skip gates)

Each gate is cheap and must PASS before the next. If a gate fails, STOP and
send me the indicated output — nothing expensive has been spent yet.
Total cost if everything passes: ~CHF 4–7 for the whole 4-model sweep.

---

## GATE 0 — upload (free)

From Git Bash on the laptop, in the folder containing `pythia_code/`:

    scp -r pythia_code georgaki@kuma.hpc.epfl.ch:~/

Then on Kuma:

    cd ~/pythia_code

PASS: `ls` shows all 15 files.

---

## GATE 1 — environment (free)

    bash setup_env.sh

PASS looks like: versions printed for python / torch / transformers /
datasets / scipy, and the line `Done. venv: /scratch/...`.
(`cuda build: 12.x` refers to the wheel; `is_available()` False here is normal.)

FAIL mode: pip network errors → the login node has no outbound internet. Rerun:

    srun --partition=l40s --gpus=1 --time=00:40:00 --pty bash setup_env.sh

(l40s is the cheap partition: CHF 0.2141/h → ~CHF 0.15 worst case.)
If it still fails → send me the last 30 lines.

---

## GATE 2 — integration test (free, ~1 min, no network)

    source /scratch/$USER/pythia_venv/bin/activate
    python test_integration.py

PASS: `ALL CHECKS PASSED` and exit without traceback. This proves the entire
pipeline (hook insertion, barycenter, gate selection, save/load) works with
the exact torch/transformers versions Gate 1 installed.

FAIL: send me the full output — this is the gate designed to catch any
version-specific incompatibility for free.

---

## GATE 3 — prefetch models + datasets (free on login node)

    export HF_HOME=/scratch/$USER/hf_cache
    python pythia_prefetch.py

PASS: five `[model] ... cached at ...` lines, two `[dataset] ... rows` lines
(pile-10k: 10,000 rows; wikitext validation: ~3,760 rows), and a total cache
size of roughly 15–20 GB.

FAIL mode: connection errors on the login node → run inside srun (l40s, as in
Gate 1). If HF itself errors (403/404/rate limit) → send me the message.

---

## GATE 4 — smoke test on GPU (~2 min, ~CHF 0.02)

    srun --partition=h100 --gpus=1 --time=00:20:00 --pty bash
    source /scratch/$USER/pythia_venv/bin/activate
    export HF_HOME=/scratch/$USER/hf_cache
    cd ~/pythia_code
    python pythia_select.py --smoke --out /tmp/smoke.json
    python pythia_confirm.py --selection /tmp/smoke.json --out /tmp/smoke_conf.json
    exit

PASS: both commands finish without traceback; select prints the gate-config
block, a position screen, gamma records with J values, and `SELECTED (...)`;
confirm prints the pre-registered rule and a VERDICT line. The numbers are
meaningless (tiny pools) — only completion matters. `gamma*=0 at selection`
is a perfectly valid PASS here.

FAIL: send me the last ~40 lines. This is where a transformers-version hook
quirk would show up on real pretrained weights — still at ~2 cents.

---

## GATE 5 — one real model (~45 min, ~CHF 0.4)

    mkdir -p logs results
    sbatch --job-name=p1b slurm_pythia.sh EleutherAI/pythia-1b
    squeue -u $USER                # wait until it disappears
    tail -f logs/p1b_*.out         # Ctrl+C to stop following

PASS checklist on the log:
  - `[base] SELECTION loss` between roughly 2.0 and 3.5 nats/token on the Pile
    (a wildly different number would mean a data/tokenizer problem);
  - Stage A prints all 15 positions; Stage B prints J for every gamma;
  - `########## STAGE 2` runs and ends with a VERDICT line;
  - `results/sel_pythia-1b_pile.json` and `results/conf_pythia-1b_pile.json` exist.

Any VERDICT is a PASS — including "NOT CONFIRMED". The gate tests the
machinery, not the hypothesis.

FAIL: send me the full log file.

---

## GATE 6 — the sweep (~3–5 h wall-clock in parallel, ~CHF 3–6)

    bash submit_all.sh
    squeue -u $USER

Jobs are independent; a failure in one does not affect the others.
When all four finish:

    python summarize.py

Send me the summarize table + the four `results/conf_*.json`. That table is
the scaling-trend result — the input to the venue decision (NeurIPS vs
TMLR/ICLR) and to updating Section 5 of the paper.

---

## Monitoring / control

    squeue -u $USER                    # queue state
    tail -f logs/<name>_<jobid>.out    # live log
    scancel <jobid>                    # kill one job
    scancel -u $USER                   # kill everything
    sausage quotas --unit GiB          # spending check (per the Kuma MOTD)

## Known-acceptable warnings (not failures)

  - `cuda available: False` in Gates 1–3 (no GPU on login node).
  - HF "resolving/downloading" progress bars in Gate 3–4.
  - A one-time tokenizer parallelism notice.

Anything that says `Traceback`, `CUDA out of memory`, `NaN`, or kills the job
is a real failure → send it to me with the gate number.
