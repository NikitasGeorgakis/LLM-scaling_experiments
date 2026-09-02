"""
otdepth.py -- shared library for training-free gated depth-insertion experiments.

Conventions follow the recorded runs (full_scale_*.json):
  * insertion index i is the 0-indexed left block; the human-readable pair is
    (F_{i+1}, F_{i+2}) in 1-indexed notation, stored as pair=[i+1, i+2].
  * gate: G_{m,gamma}(h) = h + gamma * (G_m(h) - h); gamma=0 is the exact base.
  * selection pool: 25 paired batches of 8x1024 tokens; confirmation: 40.
  * bootstrap: paired batch bootstrap, 2000 (selection) / 10000 (confirmation).
  * stability tolerances: D_KL_out <= 0.05 and D_rep <= 0.05.
"""

import copy
import hashlib
import json
import math
import os
import time
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# Grids and constants (pre-registered)
# ----------------------------------------------------------------------------

SMALL_GRID = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1]
FULL_GRID = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.2, 0.5, 1.0]
REFINE_GRID = [0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]
EPS_KL = 0.05
EPS_REP = 0.05
EPS_EFF = 0.10

def _paper_gate_config():
    """The single authoritative source for the J-objective constants
    (lambda_*, eps_*, delta, the gate grid) is otbli.config.GateConfig,
    already validated on this cluster. Importing it (rather than keeping a
    second hardcoded copy here) means the two packages cannot silently drift
    apart on what gamma* means."""
    import os
    import sys
    sys.path.insert(0, os.path.expanduser("~/otbli"))
    from otbli.config import GateConfig
    return GateConfig()
BATCH_SEQS = 8
SEQ_LEN = 1024

MODELS = {
    "gpt2-large": "openai-community/gpt2-large",
    "tinyllama": "TinyLlama/TinyLlama_v1.1",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
    "pythia-410m": "EleutherAI/pythia-410m",
    "pythia-1b": "EleutherAI/pythia-1b",
    "pythia-1.4b": "EleutherAI/pythia-1.4b",
    "pythia-2.8b": "EleutherAI/pythia-2.8b",
}

BARY_DEFAULTS = dict(tau=0.5, eta=0.05, rounds=25, sinkhorn_iters=80,
                     cost_norm="mean")


def load_model(model_key, revision=None, dtype="float32", device="cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = MODELS.get(model_key, model_key)
    td = {"float32": torch.float32, "bfloat16": torch.bfloat16,
          "float16": torch.float16}[dtype]
    tok = AutoTokenizer.from_pretrained(name, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        name, revision=revision, torch_dtype=td, attn_implementation="eager")
    model.to(device).eval()
    model.config.use_cache = False
    return model, tok


# ----------------------------------------------------------------------------
# Architecture adapters
# ----------------------------------------------------------------------------

def arch_of(model):
    mt = model.config.model_type
    if mt == "gpt2":
        return "gpt2"
    if mt == "gpt_neox":
        return "neox"
    if mt in ("llama", "mistral"):
        return "llama"
    raise ValueError(f"unsupported model_type {mt}")


def get_blocks(model):
    a = arch_of(model)
    if a == "gpt2":
        return model.transformer.h
    if a == "neox":
        return model.gpt_neox.layers
    return model.model.layers


def set_blocks(model, blocks):
    a = arch_of(model)
    if a == "gpt2":
        model.transformer.h = blocks
    elif a == "neox":
        model.gpt_neox.layers = blocks
    else:
        model.model.layers = blocks
    model.config.num_hidden_layers = len(blocks)


def mlp_descriptors(block, arch):
    """Coupled per-unit descriptors X [n_units, d] (all float32, same device)."""
    with torch.no_grad():
        if arch == "gpt2":
            w_in = block.mlp.c_fc.weight          # [h, 4h] (Conv1D: in x out)
            b_in = block.mlp.c_fc.bias            # [4h]
            w_out = block.mlp.c_proj.weight       # [4h, h]
            return torch.cat([w_in.t(), b_in[:, None], w_out],
                             dim=1).float().detach()
        if arch == "neox":
            w_in = block.mlp.dense_h_to_4h.weight   # [4h, h]
            b_in = block.mlp.dense_h_to_4h.bias     # [4h]
            w_out = block.mlp.dense_4h_to_h.weight  # [h, 4h]
            return torch.cat([w_in, b_in[:, None], w_out.t()],
                             dim=1).float().detach()
        # llama-family gated MLP: couple gate row, up row, down column
        g = block.mlp.gate_proj.weight   # [I, h]
        u = block.mlp.up_proj.weight     # [I, h]
        d = block.mlp.down_proj.weight   # [h, I]
        return torch.cat([g, u, d.t()], dim=1).float().detach()


def write_descriptors(block, arch, Z):
    Z = Z.to(next(block.parameters()).dtype)
    with torch.no_grad():
        if arch == "gpt2":
            h = block.mlp.c_fc.weight.shape[0]
            block.mlp.c_fc.weight.copy_(Z[:, :h].t())
            block.mlp.c_fc.bias.copy_(Z[:, h])
            block.mlp.c_proj.weight.copy_(Z[:, h + 1:])
        elif arch == "neox":
            h = block.mlp.dense_h_to_4h.weight.shape[1]
            block.mlp.dense_h_to_4h.weight.copy_(Z[:, :h])
            block.mlp.dense_h_to_4h.bias.copy_(Z[:, h])
            block.mlp.dense_4h_to_h.weight.copy_(Z[:, h + 1:].t())
        else:
            h = block.mlp.gate_proj.weight.shape[1]
            block.mlp.gate_proj.weight.copy_(Z[:, :h])
            block.mlp.up_proj.weight.copy_(Z[:, h:2 * h])
            block.mlp.down_proj.weight.copy_(Z[:, 2 * h:].t())


# ----------------------------------------------------------------------------
# Optimal transport
# ----------------------------------------------------------------------------

def _sqdist(X, Y):
    x2 = (X * X).sum(1, keepdim=True)
    y2 = (Y * Y).sum(1, keepdim=True).t()
    C = x2 + y2 - 2.0 * (X @ Y.t())
    return C.clamp_min_(0)


def sinkhorn_plan(X, Y, eta, iters, cost_norm="mean"):
    """Entropic OT plan between uniform measures on X and Y (log domain)."""
    C = _sqdist(X, Y)
    if cost_norm == "mean":
        C = C / C.mean().clamp_min(1e-12)
    n, m = C.shape
    loga = -math.log(n)
    logb = -math.log(m)
    f = torch.zeros(n, device=C.device)
    g = torch.zeros(m, device=C.device)
    for _ in range(iters):
        f = eta * (loga - torch.logsumexp((g[None, :] - C) / eta, dim=1))
        g = eta * (logb - torch.logsumexp((f[:, None] - C) / eta, dim=0))
    return torch.exp((f[:, None] + g[None, :] - C) / eta)


def free_support_barycenter(X1, X2, tau=0.5, eta=0.05, rounds=25,
                            sinkhorn_iters=80, cost_norm="mean"):
    """Two-measure free-support W2 barycenter, uniform weights, support n=len(X1).
    Init: index-aligned midpoint. Deterministic given inputs."""
    n = X1.shape[0]
    Y = (1 - tau) * X1 + tau * X2
    for _ in range(rounds):
        P1 = sinkhorn_plan(X1, Y, eta, sinkhorn_iters, cost_norm)
        P2 = sinkhorn_plan(X2, Y, eta, sinkhorn_iters, cost_norm)
        Y = n * ((1 - tau) * (P1.t() @ X1) + tau * (P2.t() @ X2))
    return Y


def hard_ot_match(X1, X2):
    """Minimum-cost one-to-one assignment sigma: unit k of X1 -> sigma[k] of X2."""
    from scipy.optimize import linear_sum_assignment
    C = _sqdist(X1, X2).cpu().numpy()
    r, c = linear_sum_assignment(C)
    sigma = np.empty(len(r), dtype=np.int64)
    sigma[r] = c
    return torch.as_tensor(sigma, device=X1.device)


def matching_diagnostic(X1, X2, max_units=4096, seed=0):
    """Reproduces the recorded matching diagnostic (unit cap 4096)."""
    n = X1.shape[0]
    if n > max_units:
        g = torch.Generator(device="cpu").manual_seed(seed)
        idx = torch.randperm(n, generator=g)[:max_units].to(X1.device)
        X1, X2 = X1[idx], X2[idx]
    sigma = hard_ot_match(X1, X2)
    ci = ((X1 - X2) ** 2).sum(1).mean().item()
    co = ((X1 - X2[sigma]) ** 2).sum(1).mean().item()
    frac = (sigma != torch.arange(len(sigma), device=sigma.device)).float().mean().item()
    return {"units": int(X1.shape[0]), "frac_rematched": frac,
            "mean_cost_identity": ci, "mean_cost_optimal": co,
            "reduction": 1.0 - co / ci if ci > 0 else 0.0}


# ----------------------------------------------------------------------------
# Candidate constructions
# ----------------------------------------------------------------------------

def _interp_all(cand, blk_i, blk_j, tau):
    with torch.no_grad():
        for pc, pi, pj in zip(cand.parameters(), blk_i.parameters(),
                              blk_j.parameters()):
            pc.copy_((1 - tau) * pi + tau * pj)
        for bc, bi, bj in zip(cand.buffers(), blk_i.buffers(), blk_j.buffers()):
            if bc.is_floating_point():
                bc.copy_((1 - tau) * bi.float() + tau * bj.float())


def _fresh_block(model, i):
    a = arch_of(model)
    cfg = model.config
    if a == "gpt2":
        from transformers.models.gpt2.modeling_gpt2 import GPT2Block
        cls, kw = GPT2Block, {"layer_idx": i + 1}
    elif a == "neox":
        from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer
        cls, kw = GPTNeoXLayer, {"layer_idx": i + 1}
    else:
        if cfg.model_type == "mistral":
            from transformers.models.mistral.modeling_mistral import \
                MistralDecoderLayer as cls
        else:
            from transformers.models.llama.modeling_llama import \
                LlamaDecoderLayer as cls
        kw = {"layer_idx": i + 1}
    try:
        blk = cls(cfg, **kw)
    except TypeError:
        blk = cls(cfg)
    return blk


def build_candidate(model, i, construction, tau=0.5, seed=0,
                    bary_kwargs=None, verbose=False):
    """Returns a frozen candidate block G_m for insertion after block i."""
    arch = arch_of(model)
    blocks = get_blocks(model)
    blk_i, blk_j = blocks[i], blocks[i + 1]
    dev = next(model.parameters()).device
    if construction == "copy_prev":
        return copy.deepcopy(blk_i)
    if construction == "copy_next":
        return copy.deepcopy(blk_j)
    if construction == "random":
        torch.manual_seed(seed)
        cand = _fresh_block(model, i)
        for m in cand.modules():
            model._init_weights(m)
        return cand.to(dev, next(model.parameters()).dtype)
    cand = copy.deepcopy(blk_i)
    _interp_all(cand, blk_i, blk_j, tau)      # all families interpolated
    if construction == "naive":
        return cand
    X1 = mlp_descriptors(blk_i, arch)
    X2 = mlp_descriptors(blk_j, arch)
    if construction == "hard_ot":
        sigma = hard_ot_match(X1, X2)
        Z = (1 - tau) * X1 + tau * X2[sigma]
    elif construction == "barycenter":
        bk = dict(BARY_DEFAULTS)
        bk.update(bary_kwargs or {})
        bk.pop("tau", None)
        t0 = time.time()
        Z = free_support_barycenter(X1, X2, tau=tau, **bk)
        if verbose:
            print(f"    barycenter i={i}: {time.time()-t0:.1f}s")
    else:
        raise ValueError(construction)
    write_descriptors(cand, arch, Z)
    return cand


# ----------------------------------------------------------------------------
# Gated insertion
# ----------------------------------------------------------------------------

class GatedBlock(nn.Module):
    def __init__(self, block, gamma=0.0):
        super().__init__()
        self.block = block
        self.gamma = float(gamma)
        self.record = False
        self.drift_num = 0.0
        self.drift_den = 0.0

    def reset_probe(self):
        self.drift_num = 0.0
        self.drift_den = 0.0

    def forward(self, hidden_states, *args, **kwargs):
        out = self.block(hidden_states, *args, **kwargs)
        if isinstance(out, tuple):
            h_new, rest = out[0], out[1:]
        else:
            h_new, rest = out, None
        delta = h_new - hidden_states
        mixed = hidden_states + self.gamma * delta
        if self.record:
            with torch.no_grad():
                self.drift_num += (self.gamma * delta).float().pow(2).sum().item()
                self.drift_den += hidden_states.float().pow(2).sum().item() + 1e-12
        if rest is None:
            return mixed
        return (mixed,) + rest


@contextmanager
def inserted(model, i, cand_block, gamma=0.0):
    blocks = get_blocks(model)
    n0 = len(blocks)
    gate = GatedBlock(cand_block, gamma)
    new = list(blocks)
    new.insert(i + 1, gate)
    set_blocks(model, nn.ModuleList(new))
    try:
        yield gate
    finally:
        set_blocks(model, nn.ModuleList(list(get_blocks(model))[:i + 1]
                                        + list(get_blocks(model))[i + 2:]))
        assert len(get_blocks(model)) == n0


# ----------------------------------------------------------------------------
# Pools, evaluation, statistics
# ----------------------------------------------------------------------------

def pack_pool(jsonl_path, tokenizer, n_batches, batch_seqs=None,
              seq_len=None):
    """Tokenize documents in file order, concat with EOS, pack into
    [n_batches, batch_seqs, seq_len] int64. Deterministic."""
    batch_seqs = batch_seqs or BATCH_SEQS
    seq_len = seq_len or SEQ_LEN
    need = n_batches * batch_seqs * seq_len
    eos = tokenizer.eos_token_id
    ids = []
    with open(jsonl_path) as f:
        for line in f:
            if not line.strip():
                continue
            text = json.loads(line)["text"]
            ids.extend(tokenizer(text, add_special_tokens=False).input_ids)
            ids.append(eos)
            if len(ids) >= need:
                break
    if len(ids) < need:
        raise RuntimeError(f"pool {jsonl_path}: {len(ids)} tokens < {need};"
                           " add documents to the pool")
    t = torch.tensor(ids[:need], dtype=torch.long)
    return t.view(n_batches, batch_seqs, seq_len)


@torch.no_grad()
def batch_nll(model, batch, device):
    x = batch.to(device)
    logits = model(x).logits.float()
    lp = F.log_softmax(logits[:, :-1], dim=-1)
    nll = -lp.gather(-1, x[:, 1:, None]).squeeze(-1)
    return nll.mean().item(), logits


@torch.no_grad()
def base_pass(model, pool, device, kl_probe_batches=1):
    """Per-batch base losses + cached base log-probs on probe batches (cpu fp32)."""
    losses, probes = [], []
    for b in range(pool.shape[0]):
        loss, logits = batch_nll(model, pool[b], device)
        losses.append(loss)
        if b < kl_probe_batches:
            probes.append(F.log_softmax(logits, dim=-1).cpu())
        del logits
    return np.array(losses, dtype=np.float64), probes


@torch.no_grad()
def kl_vs_base(logits, base_logp):
    lp = F.log_softmax(logits, dim=-1)
    p0 = base_logp.to(logits.device)
    kl = (p0.exp() * (p0 - lp)).sum(-1)
    return kl.mean().item()


@torch.no_grad()
def _efficiency_cost(model, cand, gate, pool, device, n_batches=5, n_repeats=3):
    """C_eff, eq. (per omega=(1/4,1/4,1/4,1/4)): FLOPs/mem/params fraction
    added by the candidate block (proxied by its own parameter count over the
    full model's), plus a REAL wall-clock latency comparison. Computed once --
    the extra block runs in full whenever gamma > 0, regardless of its value."""
    import time

    import torch

    n_cand = sum(p.numel() for p in cand.parameters())
    n_model = sum(p.numel() for p in model.parameters())
    param_frac = n_cand / max(n_model, 1)

    batches = [pool[b] for b in range(min(n_batches, pool.shape[0]))]
    def timed(gamma):
        gate.gamma = float(gamma)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_repeats):
                for b in batches:
                    model(input_ids=b.to(device), use_cache=False)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        return time.perf_counter() - t0

    t0, t1 = timed(0.0), timed(1.0)
    gate.gamma = 0.0
    lat_frac = max(t1 - t0, 0.0) / max(t0, 1e-9)
    omega = (0.25, 0.25, 0.25, 0.25)   # (FLOPs, latency, memory, params)
    return omega[0] * param_frac + omega[1] * lat_frac + omega[2] * param_frac + omega[3] * param_frac


def eval_candidate(model, pool, base_losses, base_probes, i, cand, gammas,
                   device, kl_probe_batches=1):
    """Returns records dict {gamma: {...}} and per-batch diffs [n_gamma, n_batch]."""
    nb = pool.shape[0]
    diffs = np.zeros((len(gammas), nb))
    records = {}
    with inserted(model, i, cand, 0.0) as gate:
        c_eff = _efficiency_cost(model, cand, gate, pool, device)
        for gi, g in enumerate(gammas):
            if g == 0.0:
                records["0.0"] = {"gamma": 0.0, "loss": float(base_losses.mean()),
                                  "dL_raw": 0.0, "KL": 0.0, "rep": 0.0, "Ceff": 0.0,
                                  "feasible": True}
                continue
            gate.gamma = float(g)
            gate.reset_probe()
            kls, losses = [], []
            for b in range(nb):
                gate.record = b < kl_probe_batches
                loss, logits = batch_nll(model, pool[b], device)
                losses.append(loss)
                if b < kl_probe_batches:
                    kls.append(kl_vs_base(logits, base_probes[b]))
                del logits
                gate.record = False
            losses = np.array(losses)
            diffs[gi] = losses - base_losses
            rep = gate.drift_num / max(gate.drift_den, 1e-12)
            kl = float(np.mean(kls))
            dl_mean = float(diffs[gi].mean())
            records[f"{g}"] = {
                "gamma": float(g), "loss": float(losses.mean()),
                "dL_raw": dl_mean, "KL": kl, "rep": rep, "Ceff": c_eff,
                "feasible": bool(dl_mean <= 0.0
                                and kl <= EPS_KL and rep <= EPS_REP
                                and c_eff <= EPS_EFF)}
    return records, diffs


def select_gamma(records):
    """Retained gate gamma*, matching otbli's protocol EXACTLY (not a
    loss-only proxy): argmin of the full multi-term objective J (eq. 3.41)
    over feasible gammas including 0, smallest-gamma tie-break (eq. 3.43),
    then the materiality check (eq. 3.47) -- reject back to gamma*=0 unless
    the retained point clears delta in raw loss improvement. Constants come
    from otbli.config.GateConfig, the single authoritative source (see
    _paper_gate_config above), so this cannot silently diverge from otbli's
    own already-validated numbers.
    """
    cfg = _paper_gate_config()
    L0 = records.get("0.0", {}).get("loss", None)
    eps_L = abs(L0) + cfg.eps_L if L0 is not None else cfg.eps_L

    def J(r):
        g = r["gamma"]
        return (cfg.lam_L * (r["dL_raw"] / eps_L)
                + cfg.lam_KL * r["KL"] / cfg.eps_KL
                + cfg.lam_rep * r["rep"] / cfg.eps_rep
                + cfg.lam_eff * r["Ceff"] / cfg.eps_eff
                + cfg.lam_gamma * g * g)

    feas = [r for r in records.values() if r["feasible"]]
    if not feas:
        return 0.0, 0.0
    j_vals = {r["gamma"]: J(r) for r in feas}
    j_min = min(j_vals.values())
    g_hat = min(g for g, j in j_vals.items() if j == j_min)     # eq. (3.43)
    r_hat = next(r for r in feas if r["gamma"] == g_hat)

    if g_hat > 0.0 and r_hat["dL_raw"] < -cfg.delta:            # eq. (3.47)
        return g_hat, r_hat["dL_raw"]
    return 0.0, 0.0


def boot_ci(d, n_boot, seed=0, lo=2.5, hi=97.5):
    d = np.asarray(d, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return [float(np.percentile(means, lo)), float(np.percentile(means, hi))]


def one_sided_t(d):
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    se = d.std(ddof=1) / math.sqrt(n)
    if se == 0:
        return 0.0, 0.5
    t = d.mean() / se
    try:
        from scipy import stats
        p = float(stats.t.cdf(t, df=n - 1))
    except Exception:
        p = float(0.5 * math.erfc(-t / math.sqrt(2)))  # normal approx
    return float(t), p


def check_exact_recovery(model, pool, i, cand, device):
    with torch.no_grad():
        base = model(pool[0].to(device)).logits
        with inserted(model, i, cand, 0.0):
            ins = model(pool[0].to(device)).logits
    return bool(torch.equal(base, ins)), float((base - ins).abs().max().item())


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_locked(path, model_key, revision, i, construction, gamma, tau, seed,
                cand, extra=None):
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model_key": model_key, "revision": revision, "i": int(i),
                "pair": [int(i) + 1, int(i) + 2], "construction": construction,
                "gamma": float(gamma), "tau": float(tau), "seed": int(seed),
                "extra": extra or {},
                "state_dict": {k: v.cpu() for k, v in
                               cand.state_dict().items()}}, path)


def load_locked(path, model):
    meta = torch.load(path, map_location="cpu", weights_only=False)
    cand = copy.deepcopy(get_blocks(model)[meta["i"]])
    cand.load_state_dict(meta["state_dict"])
    cand.to(next(model.parameters()).device, next(model.parameters()).dtype)
    return meta, cand
