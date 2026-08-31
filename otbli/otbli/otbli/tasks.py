"""Downstream-task evaluation with per-QUESTION scores, so that a gate setting
can be compared to gamma = 0 by exact pairing (same task, same doc_id, same
few-shot context) rather than by comparing two aggregate numbers.

Backend: lm-evaluation-harness. The harness wraps the ALREADY-LOADED model
object, so the gate hook installed by otbli.insertion is live during scoring:
flipping gamma between calls changes what the harness measures without
reloading or re-tokenizing anything.

Only likelihood-scored tasks (multiple choice / cloze) are supported: the gate
is defined for teacher-forced evaluation (use_cache=False), and free
generation with an active gate is out of scope (Section 6.1, note (ii)).
"""
import inspect

import numpy as np


def _lm_eval():
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError as e:                                   # pragma: no cover
        raise ImportError(
            "the task experiment needs lm-evaluation-harness:\n"
            "    pip install 'lm-eval>=0.4.2'"
        ) from e
    return lm_eval, HFLM


def build_lm(model, tokenizer, batch_size: int = 8, device: str = "cuda"):
    """Wrap an already-loaded model (with its gate hook) for the harness."""
    _, HFLM = _lm_eval()
    return HFLM(pretrained=model, tokenizer=tokenizer,
                batch_size=batch_size, device=device)


def list_tasks(pattern: str = "") -> list:
    """Task names available in the installed harness, optionally filtered."""
    from lm_eval.tasks import TaskManager
    names = sorted(TaskManager().all_tasks)
    return [n for n in names if pattern in n] if pattern else names


# preference order; whichever is present is used, and pinned per task so the
# same metric is compared across gammas
_METRIC_KEYS = ("acc_norm", "acc", "exact_match", "em")


def _pick_metric(sample: dict) -> str:
    for k in _METRIC_KEYS:
        if k in sample and isinstance(sample[k], (int, float, bool)):
            return k
    raise KeyError(f"no supported accuracy metric in sample keys {list(sample)}; "
                   f"supported: {_METRIC_KEYS}")


def eval_tasks_per_doc(lm, tasks, limit=None, num_fewshot=0, seed: int = 1234,
                       metrics: dict = None) -> dict:
    """Run the harness and return {task: {"doc_ids": [...], "scores": [...],
    "metric": name}} with one score per question.

    All seeds are pinned so the document order and the few-shot contexts are
    identical across calls — this is what makes the comparison paired.
    """
    lm_eval, _ = _lm_eval()
    kw = dict(model=lm, tasks=list(tasks), limit=limit, num_fewshot=num_fewshot,
              log_samples=True, bootstrap_iters=0, cache_requests=False,
              random_seed=seed, numpy_random_seed=seed, torch_random_seed=seed,
              fewshot_random_seed=seed, verbosity="ERROR")
    accepted = set(inspect.signature(lm_eval.simple_evaluate).parameters)
    out = lm_eval.simple_evaluate(**{k: v for k, v in kw.items() if k in accepted})

    per_task = {}
    for task, samples in out["samples"].items():
        samples = sorted(samples, key=lambda s: s["doc_id"])
        key = (metrics or {}).get(task) or _pick_metric(samples[0])
        per_task[task] = {
            "doc_ids": [int(s["doc_id"]) for s in samples],
            "scores": np.array([float(s[key]) for s in samples], dtype=np.float64),
            "metric": key,
        }
    return per_task


def flatten(per_task: dict):
    """Flatten to (keys, scores) with keys = [(task, doc_id), ...] so that two
    evaluations can be aligned question by question."""
    keys, vals = [], []
    for task in sorted(per_task):
        d = per_task[task]
        for doc_id, s in zip(d["doc_ids"], d["scores"]):
            keys.append((task, doc_id))
            vals.append(s)
    return keys, np.asarray(vals, dtype=np.float64)


def align(keys_ref, per_task: dict) -> np.ndarray:
    """Scores of `per_task` in the order of `keys_ref`; raises if the question
    set moved (which would silently break pairing)."""
    keys, vals = flatten(per_task)
    if keys != keys_ref:
        raise RuntimeError(
            "question set changed between gate settings — pairing is invalid. "
            "Check that seeds/limit/num_fewshot are identical across calls.")
    return vals


def split_questions(keys, seed: int = 1234, frac_sel: float = 0.5):
    """Stratified per-task split of question indices into a selection half and
    a disjoint confirmation half (the task analogue of the two Pile pools)."""
    rng = np.random.default_rng(seed)
    by_task = {}
    for idx, (task, _) in enumerate(keys):
        by_task.setdefault(task, []).append(idx)
    sel, conf = [], []
    for task in sorted(by_task):
        idx = np.array(by_task[task])
        perm = rng.permutation(len(idx))
        cut = int(round(frac_sel * len(idx)))
        sel.extend(idx[perm[:cut]].tolist())
        conf.extend(idx[perm[cut:]].tolist())
    return np.array(sorted(sel)), np.array(sorted(conf))


def mcnemar_exact(d: np.ndarray):
    """Exact McNemar test on paired binary outcomes, the natural companion to
    the bootstrap for 0/1 scores. d = score(gamma) - score(0) per question:
    c = #(gained), b = #(lost), discordant pairs n = b + c.
    Returns (b, c, two-sided p)."""
    from scipy import stats
    d = np.asarray(d)
    c = int((d > 0).sum())
    b = int((d < 0).sum())
    if b + c == 0:
        return b, c, 1.0
    return b, c, float(stats.binomtest(c, b + c, 0.5).pvalue)
