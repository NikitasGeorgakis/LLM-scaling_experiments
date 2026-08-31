#!/usr/bin/env python3
"""Smoke test on tiny randomly-initialized models (CPU, no downloads).
Run this BEFORE burning H100 hours:  python test_smoke.py
Checks: insertion/removal, bit-exact gamma=0 recovery, all constructions,
drift identity D_rep(g)=g^2 D_rep(1), selection/bootstrap, locked save/load,
and the max-t analysis on synthetic records.
"""
import copy
import os
import tempfile

import numpy as np
import torch

import otdepth as od

torch.manual_seed(0)
DEV = "cpu"
TINY_BARY = dict(eta=0.05, rounds=3, sinkhorn_iters=15, cost_norm="mean")


def tiny_models():
    from transformers import (GPT2Config, GPT2LMHeadModel, GPTNeoXConfig,
                              GPTNeoXForCausalLM, LlamaConfig,
                              LlamaForCausalLM, MistralConfig,
                              MistralForCausalLM)
    out = {}
    out["gpt2"] = GPT2LMHeadModel(GPT2Config(
        n_layer=4, n_embd=32, n_head=2, vocab_size=101, n_positions=128))
    out["neox"] = GPTNeoXForCausalLM(GPTNeoXConfig(
        num_hidden_layers=4, hidden_size=32, intermediate_size=128,
        num_attention_heads=2, vocab_size=101, max_position_embeddings=128))
    out["llama"] = LlamaForCausalLM(LlamaConfig(
        num_hidden_layers=4, hidden_size=32, intermediate_size=48,
        num_attention_heads=2, num_key_value_heads=2, vocab_size=101,
        max_position_embeddings=128))
    out["mistral"] = MistralForCausalLM(MistralConfig(
        num_hidden_layers=4, hidden_size=32, intermediate_size=48,
        num_attention_heads=2, num_key_value_heads=1, vocab_size=101,
        max_position_embeddings=128, sliding_window=64))
    for m in out.values():
        m.eval()
        m.config.use_cache = False
    return out


def main():
    pool = torch.randint(0, 101, (3, 2, 64))     # 3 batches, 2 seqs, len 64
    for name, model in tiny_models().items():
        print(f"== {name} ==")
        n0 = len(od.get_blocks(model))
        base, probes = od.base_pass(model, pool, DEV, kl_probe_batches=1)
        assert np.isfinite(base).all()

        for c in ["copy_next", "copy_prev", "naive", "hard_ot", "barycenter",
                  "random"]:
            cand = od.build_candidate(model, 1, c, 0.5, seed=3,
                                      bary_kwargs=TINY_BARY)
            ok, mx = od.check_exact_recovery(model, pool, 1, cand, DEV)
            assert ok, f"{name}/{c}: gamma=0 not bit-exact (max {mx})"
            assert len(od.get_blocks(model)) == n0, "removal failed"
            rec, diffs = od.eval_candidate(model, pool, base, probes, 1,
                                           cand, [0.0, 0.1, 1.0], DEV, 1)
            assert diffs.shape == (3, 3)
            r01, r1 = rec["0.1"], rec["1.0"]
            assert np.isfinite([r01["dL_raw"], r01["KL"], r01["rep"]]).all()
            if r1["rep"] > 1e-9:
                ratio = r01["rep"] / r1["rep"]
                assert abs(ratio - 0.01) < 0.002, \
                    f"{name}/{c}: drift identity violated ({ratio:.4f})"
            print(f"  {c:<11} dL(0.1)={r01['dL_raw']:+.3e} "
                  f"rep ratio ok, gamma=0 exact")
            del cand

        # selection + bootstrap + locked save/load roundtrip
        cand = od.build_candidate(model, 0, "naive", 0.5)
        rec, diffs = od.eval_candidate(model, pool, base, probes, 0, cand,
                                       od.SMALL_GRID, DEV, 1)
        g, dL = od.select_gamma(rec)
        ci = od.boot_ci(diffs[od.SMALL_GRID.index(g)] if g > 0 else diffs[0],
                        200, seed=1)
        t, p = od.one_sided_t(diffs[-1])
        assert ci[0] <= ci[1] and 0 <= p <= 1
        with tempfile.TemporaryDirectory() as td:
            pth = os.path.join(td, "x.pt")
            od.save_locked(pth, name, None, 0, "naive", max(g, 0.1), 0.5, 0,
                           cand)
            meta, cand2 = od.load_locked(pth, model)
            for p1, p2 in zip(cand.parameters(), cand2.parameters()):
                assert torch.equal(p1, p2)
        print(f"  select gamma*={g}, CI/t/p sane, locked roundtrip ok")
        del cand

        # matching diagnostic runs
    print("== max-t on synthetic records ==")
    rng = np.random.default_rng(0)
    D = rng.normal(0, 1e-3, size=(6, 4, 25))
    D[2, 3] -= 4e-3                               # one true effect
    mask = np.ones((6, 4), bool)
    mask[:, 0] = False
    t_obs, mean, p_glob, p_adj, tmin = __import__("run_maxt").maxt(
        D, mask, 500, 0)
    assert p_adj[2, 3] < 0.05 and p_glob < 0.05
    null = rng.normal(0, 1e-3, size=(6, 4, 25))
    _, _, pg2, _, _ = __import__("run_maxt").maxt(null, mask, 500, 1)
    assert pg2 > 0.05
    print(f"  true cell adjusted p={p_adj[2,3]:.3f}, "
          f"null global p={pg2:.2f} -- ok")
    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
