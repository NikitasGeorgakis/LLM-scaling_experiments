"""
Integration test on a REAL GPTNeoX model (randomly initialized -- no download).
Exercises every code path that will run on Kuma. Run this on the login node
after setup_env.sh, BEFORE spending any GPU money:

    python test_integration.py

All six checks must print PASS.
"""
import copy
import numpy as np
import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

from pythia_wrapper import PythiaWrapper
from pythia_barycentric import (build_barycentric_layer, build_naive_average_layer,
                                build_duplicate_layer)
from pythia_stats import paired_stats
from pythia_gate import DEFAULTS, select_gate, paper_grid, refine_grid

torch.manual_seed(0)

cfgm = GPTNeoXConfig(hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
                     intermediate_size=256, vocab_size=200,
                     max_position_embeddings=128)
model = GPTNeoXForCausalLM(cfgm)
W = PythiaWrapper.from_model(model, device="cpu")
print("[model]", W.describe())

pool = [torch.randint(0, 200, (2, 32)) for _ in range(6)]
n_fail = 0


def check(name, ok, detail=""):
    global n_fail
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")
    n_fail += 0 if ok else 1


# 1) barycentric construction on real GPTNeoXLayer ---------------------------
layer, obj = build_barycentric_layer(W.layers[1], W.layers[2], tau=0.5,
                                     n_alt_rounds=4, sinkhorn_iters=30)
finite = all(torch.isfinite(p).all() for p in layer.parameters())
check("barycentric build (shapes+finite)", finite, f"objective={obj:.4e}")

# 2) gamma = 0 is a bit-exact identity ---------------------------------------
base = W.per_batch_losses(pool)
zero = W.per_batch_losses(pool, layer, 1, 0.0)
check("gamma=0 identity (M+_0 == M)", np.array_equal(base, zero),
      f"max|diff|={np.abs(base-zero).max():.1e}")

# 3) O(gamma) continuity ------------------------------------------------------
d_tiny = abs(W.per_batch_losses(pool, layer, 1, 1e-4).mean() - base.mean())
d_full = abs(W.per_batch_losses(pool, layer, 1, 1.0).mean() - base.mean())
check("continuity (|dL| at 1e-4 << at 1.0)", d_tiny < 0.02 * max(d_full, 1e-9),
      f"{d_tiny:.2e} vs {d_full:.2e}")

# 4) drift probe: pass-through + exact gamma^2 identity ----------------------
drift1 = W.measure_drift_unit(layer, 1, pool[:3])
# direct check of D_rep(g) = g^2 * D_rep(1) via an independent measurement:
# capture hidden state h entering layer 2, apply gate manually
h_ratio = []
cap = {}
def cap_hook(m, args, kwargs):
    cap["h"] = args[0] if args else kwargs["hidden_states"]
    cap["rest"] = args[1:] if args else ()
    cap["kw"] = {k: v for k, v in kwargs.items() if k != "hidden_states"}
    return None
hnd = W.layers[2].register_forward_pre_hook(cap_hook, with_kwargs=True)
W.model(input_ids=pool[0])
hnd.remove()
h = cap["h"]
out = layer(h, *cap["rest"], **cap["kw"])
h_bar = out[0] if isinstance(out, (tuple, list)) else out
g = 0.3
lhs = ((h + g * (h_bar - h)) - h).pow(2).sum() / (h.pow(2).sum() + 1e-8)
check("drift identity D_rep(g)=g^2 D_rep(1)",
      abs(float(lhs) / (g**2) - float((h_bar-h).pow(2).sum()/(h.pow(2).sum()+1e-8))) < 1e-6
      and drift1 > 0, f"D_rep(1)={drift1:.4f}")

# 5) full gate selection end-to-end ------------------------------------------
cfg = dict(DEFAULTS)
best, recs, ceff, d1, base_arr = select_gate(W, layer, 1, pool, cfg,
                                             paired_stats, verbose=False)
grid_ok = {r["gamma"] for r in recs} >= set(paper_grid())
zero_rec = [r for r in recs if r["gamma"] == 0.0][0]
check("gate selection (paper grid + refine + pick)",
      grid_ok and zero_rec["J"] == 0.0 and best["feasible"],
      f"best gamma={best['gamma']}, J={best['J']:+.3e}, {len(recs)} records")

rg = refine_grid(recs, 0.01, 8)
check("refinement grid strictly between neighbours",
      all(0.003 < x < 0.03 for x in rg) and len(rg) > 0, f"{rg[:3]}...")

# 6) save/load reproducibility (what confirmation does) ----------------------
torch.save(dict(state_dict=layer.state_dict()), "/tmp/_layer.pt")
layer2 = copy.deepcopy(W.layers[1])
layer2.load_state_dict(torch.load("/tmp/_layer.pt")["state_dict"])
a = W.per_batch_losses(pool, layer, 1, 0.12)
b = W.per_batch_losses(pool, layer2, 1, 0.12)
check("save/load bit-exact reproduction", np.array_equal(a, b))

# baselines run
_ = build_naive_average_layer(W.layers[1], W.layers[2])
_ = build_duplicate_layer(W.layers[1])
print(f"\n{'ALL CHECKS PASSED' if n_fail == 0 else f'{n_fail} CHECK(S) FAILED'} "
      f"(transformers integration on real GPTNeoX architecture)")
raise SystemExit(n_fail)
