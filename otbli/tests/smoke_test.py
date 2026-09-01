#!/usr/bin/env python3
"""Offline end-to-end integration test on a tiny RANDOM GPT-NeoX model.

Requires no downloads: the model is instantiated from a config. Checks that
the machinery behaves exactly as Section 3 specifies:
  1. Sinkhorn marginals are respected and the barycenter objective decreases.
  2. Corollary 3.1 sanity: barycenter pairing cost <= identity-pairing cost.
  3. Stage A + Stage B run end to end and select gamma* in [0, 1].
  4. Exact recovery: hook-free logits are bit-identical before/after all
     insertion machinery, and no parameter is ever modified.
  5. Drift identity D_rep(gamma) = gamma^2 D_rep(1) to numerical precision.
  6. Baselines and the exact-LAP matching diagnostic run.
  7. A confirmation call executes rule (3.48) on the disjoint pool.

Run:  python tests/smoke_test.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from otbli.config import OTConfig, GateConfig, ProtocolConfig
from otbli.sinkhorn import pairwise_sq_dists, sinkhorn_log
from otbli.barycenter import free_support_barycenter
from otbli.protocol import (stage_a_screen, stage_b_gate, baseline_deltas,
                            loss_only_candidate, run_confirmation,
                            make_insertion)
from otbli.metrics import batch_losses
from otbli.diagnostics import (matching_diagnostic, state_fingerprint,
                               fingerprints_equal, exact_recovery_check,
                               drift_identity_check)


def check(name, ok):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    assert ok, name


def main():
    torch.manual_seed(0)
    device = "cpu"

    # ---------------------------------------------------------- 1. Sinkhorn
    X = torch.randn(48, 17)
    Y = torch.randn(48, 17) + 0.5
    a = torch.full((48,), 1 / 48.0)
    C = pairwise_sq_dists(X, Y)
    P = sinkhorn_log(a, a, C, eta=0.05, n_iters=200)
    check("Sinkhorn row marginals", torch.allclose(P.sum(1), a, atol=1e-4))
    check("Sinkhorn col marginals", torch.allclose(P.sum(0), a, atol=1e-4))

    Z = free_support_barycenter(X, Y, tau=0.5, eta=0.05, n_rounds=10, n_sinkhorn=100)
    cost_bary = 0.5 * pairwise_sq_dists(X, Z).min(1).values.mean() \
        + 0.5 * pairwise_sq_dists(Y, Z).min(1).values.mean()
    cost_id = 0.5 * ((X - ((X + Y) / 2)) ** 2).sum(1).mean() \
        + 0.5 * ((Y - ((X + Y) / 2)) ** 2).sum(1).mean()
    check("barycenter no worse than identity pairing (Cor. 3.1)",
          cost_bary <= cost_id + 1e-6)

    # ------------------------------------------------- 2. tiny random model
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
    cfg = GPTNeoXConfig(vocab_size=128, hidden_size=32, num_hidden_layers=4,
                        num_attention_heads=4, intermediate_size=128,
                        max_position_embeddings=128)
    model = GPTNeoXForCausalLM(cfg).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    g = torch.Generator().manual_seed(1)
    sel = torch.randint(0, 128, (4, 2, 32), generator=g)
    conf = torch.randint(0, 128, (5, 2, 32), generator=g)

    fp0 = state_fingerprint(model)
    ref = exact_recovery_check(model, sel[0], device)

    ot_cfg = OTConfig(n_alt_rounds=5, n_sinkhorn_iters=40)
    gate_cfg = GateConfig(refine_Q=4)
    proto = ProtocolConfig(n_boot_sel=200, n_boot_conf=500, kl_batches=2,
                           drift_batches=2, latency_batches=2,
                           latency_repeats=1, match_max_units=64)

    base_sel = batch_losses(model, sel, device)
    screen = stage_a_screen(model, sel, base_sel, device, ot_cfg, proto,
                            verbose=False)
    check("Stage A screened all L-1 positions", len(screen) == 3)

    res = stage_b_gate(model, sel, base_sel, device, screen[0]["i"], ot_cfg,
                       gate_cfg, proto, verbose=False)
    check("gamma* in [0,1]", 0.0 <= res["gamma_star"] <= 1.0)
    check("gamma=0 record is feasible with J=0",
          res["records"][0.0]["feasible"] and res["records"][0.0]["J"] == 0.0)
    check("safe fallback semantics: gamma*=0 unless materially better",
          res["gamma_star"] == 0.0
          or res["records"][res["gamma_star"]]["loss"] < res["L0"] - gate_cfg.delta)

    # ------------------------------------------------------- 3. diagnostics
    i = screen[0]["i"]
    ins = make_insertion(model, i, ot_cfg, device)
    dic = drift_identity_check(model, ins, sel, device, res["drift1"],
                               gamma=0.1, n_batches=2)
    check(f"drift identity (rel err {dic['rel_err']:.1e})", dic["rel_err"] < 1e-4)
    ins.remove()

    md = matching_diagnostic(model.gpt_neox.layers[i],
                             model.gpt_neox.layers[i + 1], max_units=64)
    check("matching diagnostic: optimal <= identity cost",
          md["mean_cost_optimal"] <= md["mean_cost_identity"] + 1e-9)

    bl = baseline_deltas(model, sel, base_sel, device, i, ot_cfg.tau, n_boot=200)
    check("baselines computed", set(bl) == {"naive_average", "duplicate"})

    cand = loss_only_candidate([res], gate_cfg, n_boot=200)
    check("loss-only rule runs (candidate or None)",
          cand is None or cand["gamma"] > 0)

    conf_res = run_confirmation(model, conf, device, i, 0.01, ot_cfg, gate_cfg,
                                n_boot=500)
    check("confirmation returns rule-(3.48) fields",
          {"d_mean", "ci", "t", "p_one_sided", "accepted"} <= set(conf_res))

    # ---------------------------------------------------- 4. exact recovery
    check("weights untouched after entire pipeline",
          fingerprints_equal(fp0, state_fingerprint(model)))
    check("hook-free logits bit-identical (M+_0 == M)",
          exact_recovery_check(model, sel[0], device, ref))


    # ------------------------------------- 5. other architectures, end to end
    from otbli.atomize import mlp_atoms, write_mlp_atoms, _detect_mlp_type
    import copy as _copy

    from transformers import GPT2Config, GPT2LMHeadModel
    from transformers import LlamaConfig, LlamaForCausalLM
    tiny = {
        "gpt2": GPT2LMHeadModel(GPT2Config(vocab_size=128, n_positions=64,
                                           n_embd=32, n_layer=3, n_head=4)),
        "gated": LlamaForCausalLM(LlamaConfig(vocab_size=128, hidden_size=32,
                                              intermediate_size=64,
                                              num_hidden_layers=3,
                                              num_attention_heads=4,
                                              num_key_value_heads=4,
                                              max_position_embeddings=64)),
    }
    from otbli.arch import get_layers
    for arch_name, m2 in tiny.items():
        m2 = m2.to(device).eval()
        for p in m2.parameters():
            p.requires_grad_(False)
        lyr = get_layers(m2)
        check(f"[{arch_name}] detected", _detect_mlp_type(lyr[0]) == arch_name)

        # Q_b o A_b = id in the (gauge-fixed) atom coordinates
        A0 = mlp_atoms(lyr[0], gauge_fix=True)
        probe = _copy.deepcopy(lyr[0])
        write_mlp_atoms(probe, A0)
        check(f"[{arch_name}] atom write/read round-trip",
              torch.allclose(mlp_atoms(probe, gauge_fix=True), A0, atol=1e-5))

        sel2 = torch.randint(0, 128, (3, 2, 32), generator=g)
        fp = state_fingerprint(m2)
        ref2 = exact_recovery_check(m2, sel2[0], device)
        base2 = batch_losses(m2, sel2, device)
        sc = stage_a_screen(m2, sel2, base2, device, ot_cfg, proto, verbose=False)
        rb = stage_b_gate(m2, sel2, base2, device, sc[0]["i"], ot_cfg, gate_cfg,
                          proto, verbose=False)
        check(f"[{arch_name}] gamma* in [0,1]", 0.0 <= rb["gamma_star"] <= 1.0)
        md2 = matching_diagnostic(lyr[sc[0]["i"]], lyr[sc[0]["i"] + 1], max_units=64)
        check(f"[{arch_name}] matching diagnostic runs",
              "reduction" in md2 and (arch_name != "gated" or "no_gauge" in md2))
        check(f"[{arch_name}] weights untouched",
              fingerprints_equal(fp, state_fingerprint(m2)))
        check(f"[{arch_name}] exact recovery",
              exact_recovery_check(m2, sel2[0], device, ref2))

    print("\nAll smoke checks passed.")


if __name__ == "__main__":
    main()
