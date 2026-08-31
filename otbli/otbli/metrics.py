"""Selection/confirmation metrics.

  * batch_losses    : per-batch autoregressive cross-entropy (nats/token), eq. (3.30)
  * batch_kl        : D_KL^out(gamma), eq. (3.32)
  * efficiency_cost : C_eff(gamma), eq. (3.34) — constant over gamma > 0 because
                      the inserted block runs in full regardless of gate value;
                      C_eff(0) = 0 (block bypassed), eq. (3.36)
  * bootstrap_ci    : nonparametric bootstrap over paired per-batch differences
  * paired_t        : one-sided paired t-test (as reported in Tables 4 and 7)
"""
import time
import numpy as np
import torch


@torch.no_grad()
def batch_losses(model, pool, device) -> np.ndarray:
    """Per-batch mean cross-entropy in nats/token. Batches are evaluated in a
    fixed order so paired differences with the base model are well defined."""
    out = []
    for batch in pool:
        ids = batch.to(device)
        loss = model(input_ids=ids, labels=ids, use_cache=False).loss
        out.append(float(loss.item()))
    return np.asarray(out, dtype=np.float64)


@torch.no_grad()
def batch_kl(model, insertion, pool, device, chunk: int = 128) -> float:
    """Mean over contexts/positions of KL( p_0(.|c) || p_gamma(.|c) )."""
    gamma = insertion.gamma
    tot, cnt = 0.0, 0
    for batch in pool:
        ids = batch.to(device)
        insertion.set_gamma(0.0)
        l0 = model(input_ids=ids, use_cache=False).logits
        insertion.set_gamma(gamma)
        lg = model(input_ids=ids, use_cache=False).logits
        for s in range(0, ids.shape[1], chunk):        # chunk over positions (vocab is big)
            a = torch.log_softmax(l0[:, s:s + chunk].float(), dim=-1)
            b = torch.log_softmax(lg[:, s:s + chunk].float(), dim=-1)
            tot += (a.exp() * (a - b)).sum(dim=-1).sum().item()
            cnt += a.shape[0] * a.shape[1]
        del l0, lg
    insertion.set_gamma(gamma)
    return tot / max(cnt, 1)


def _matmul_params(module) -> int:
    """Matmul parameters of every Linear (and GPT-2 Conv1D) in the module,
    LM head included when the full model is passed."""
    import torch.nn as nn
    total = 0
    for m in module.modules():
        if isinstance(m, nn.Linear) or type(m).__name__ == "Conv1D":
            total += m.weight.numel()
    return total


@torch.no_grad()
def efficiency_cost(model, gbar, insertion, pool, device, omega,
                    n_batches: int = 5, repeats: int = 3) -> dict:
    """C_eff for gamma > 0: relative overheads in FLOPs (analytic, matmul-
    dominant, incl. the LM head), latency (measured), memory and parameter
    count (both = parameter ratio for a float32 deployment)."""
    f_block = _matmul_params(gbar)
    f_model = _matmul_params(model)          # blocks + LM head, arch-agnostic
    r_flops = f_block / f_model

    p_block = sum(p.numel() for p in gbar.parameters())
    p_model = sum(p.numel() for p in model.parameters())
    r_par = p_block / p_model
    r_mem = r_par

    def timed(gamma: float) -> float:
        ts = []
        for _ in range(repeats):
            insertion.set_gamma(gamma)
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for b in pool[:n_batches]:
                model(input_ids=b.to(device), use_cache=False)
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) / max(n_batches, 1))
        return float(np.median(ts))

    t_base = timed(0.0)
    t_ins = timed(1e-4)          # any gamma > 0: the block executes in full
    insertion.set_gamma(0.0)
    r_lat = max(t_ins - t_base, 0.0) / max(t_base, 1e-12)

    wF, wT, wM, wP = omega
    return {
        "C_eff": wF * r_flops + wT * r_lat + wM * r_mem + wP * r_par,
        "flops": r_flops, "latency": r_lat, "memory": r_mem, "params": r_par,
    }


def bootstrap_ci(d: np.ndarray, n_resamples: int, seed: int = 0,
                 alpha: float = 0.05):
    """Percentile bootstrap CI for the mean of paired differences d."""
    d = np.asarray(d, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_resamples, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def paired_t(d: np.ndarray):
    """Paired t-test of mean(d) = 0; one-sided p for the improvement direction."""
    from scipy import stats
    d = np.asarray(d, dtype=np.float64)
    t, p_two = stats.ttest_1samp(d, 0.0)
    p_one = p_two / 2.0 if t < 0 else 1.0 - p_two / 2.0
    return float(t), float(p_one)
