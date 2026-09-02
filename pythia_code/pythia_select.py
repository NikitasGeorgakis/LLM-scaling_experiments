"""
STAGE 1 -- SELECTION (paper eq. 3.48, first half), full-fidelity version.

Stage A screens all insertion positions at gamma=1 (paired dL), as in
Section 5. Stage B runs the COMPLETE gate selection of Section 3.4 at the
top-k positions: paper grid (3.29), all four criteria, feasibility (3.36),
objective (3.40), smallest-minimizer tie-break (3.42) and one refinement pass
(3.43). The winning (position, gamma) minimizes J across positions.

The constructed barycentric layer of the winning position is SAVED to disk, so
confirmation loads bit-identical parameters instead of rebuilding.

Usage:
    python pythia_select.py --model EleutherAI/pythia-1b --out results/sel_1b.json
"""
import argparse, json, os, time
import numpy as np
import torch

from pythia_wrapper import PythiaWrapper
from pythia_barycentric import (build_barycentric_layer, build_naive_average_layer,
                                build_duplicate_layer, matching_diagnostics)
from pythia_data import load_token_stream, make_pools
from pythia_stats import paired_stats
from pythia_gate import DEFAULTS, print_gate_config, select_gate


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-1b")
    ap.add_argument("--dataset", default="pile", choices=["pile", "wikitext"])
    ap.add_argument("--block-size", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--sel-batches", type=int, default=25)
    ap.add_argument("--conf-batches", type=int, default=40)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--alt-rounds", type=int, default=25)
    ap.add_argument("--sinkhorn-iters", type=int, default=80)
    ap.add_argument("--max-tokens", type=int, default=2_000_000)
    ap.add_argument("--positions", default="all")
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--out", default="results/selection.json")
    ap.add_argument("--layer-out", default=None,
                    help="path for the saved winning layer (default: <out>.layer.pt)")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--revision", default=None, help="HF revision, e.g. step8000")
    ap.add_argument("--strong", action="store_true",
                    help="headline-run preset: sel=40, conf=60, top-k=3")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny fast run on pythia-70m to shake down the pipeline")
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k.replace('_','-')}", type=type(v), default=v)
    args = ap.parse_args()
    if args.strong:
        args.sel_batches, args.conf_batches, args.top_k = 40, 60, 3
    if args.smoke:
        args.model = "EleutherAI/pythia-70m"
        args.block_size, args.batch = 256, 2
        args.sel_batches, args.conf_batches = 4, 4
        args.alt_rounds, args.sinkhorn_iters = 3, 20
        args.max_tokens, args.top_k = 100_000, 1
        print(">>> SMOKE MODE: numbers meaningless; verifies the pipeline only.\n")
    return args


def main():
    args = parse()
    cfg = {k: getattr(args, k) for k in DEFAULTS}
    torch.manual_seed(2026)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    print(f"[env] device={dev}  torch={torch.__version__}")

    print_gate_config(cfg)
    print()

    W = PythiaWrapper(args.model, dtype=dtype, device=dev, cache_dir=args.cache_dir, revision=args.revision)
    print("[model]", W.describe())

    stream = load_token_stream(W.tokenizer, args.dataset, args.max_tokens, args.cache_dir)
    sel_pool, _ = make_pools(stream, args.block_size, args.batch,
                             args.sel_batches, args.conf_batches)

    t0 = time.time()
    base_sel = W.per_batch_losses(sel_pool)
    print(f"\n[base] SELECTION loss = {base_sel.mean():.6f} nats/token "
          f"(SE {base_sel.std(ddof=1)/np.sqrt(len(base_sel)):.6f})\n")

    positions = (list(range(W.n_layer - 1)) if args.positions == "all"
                 else [int(x) for x in args.positions.split(",")])

    # ---------------- Stage A: position screen (gamma=1) -------------------
    print(f"--- Stage A: position screen (gamma=1, tau={args.tau}) ---")
    screen, cache = [], {}
    for i in positions:
        tA = time.time()
        layer, obj = build_barycentric_layer(
            W.layers[i], W.layers[i + 1], tau=args.tau, eta=args.eta,
            n_alt_rounds=args.alt_rounds, sinkhorn_iters=args.sinkhorn_iters)
        cache[i] = layer
        cand = W.per_batch_losses(sel_pool, layer, i, 1.0)
        st = paired_stats(base_sel, cand, n_boot=2000)
        screen.append((i, float(st["mean"])))
        print(f"pos {i:2d} (F_{i+1},F_{i+2}): dL = {st['mean']:+.6f} "
              f"CI95=[{st['ci95'][0]:+.6f},{st['ci95'][1]:+.6f}] "
              f"({time.time()-tA:.0f}s)")
    screen.sort(key=lambda r: r[1])
    top = [p for p, _ in screen[:args.top_k]]
    print(f"\nTop-{args.top_k} positions: {top}")

    # ---------------- Stage B: full gate selection (Sec 3.4) ---------------
    print(f"\n--- Stage B: full gate selection at top positions ---")
    per_pos = {}
    for i in top:
        print(f"  position {i}:")
        best, recs, ceff, drift1, _ = select_gate(W, cache[i], i, sel_pool,
                                                  cfg, paired_stats)
        per_pos[i] = dict(best=best, records=recs, ceff=ceff, drift_unit=drift1)

    i_star = min(per_pos, key=lambda i: per_pos[i]["best"]["J"])
    best = per_pos[i_star]["best"]
    g_star = best["gamma"]
    print(f"\nSELECTED (selection set only): position={i_star}, gamma={g_star}, "
          f"tau={args.tau}  J={best['J']:+.4e}  dL={best['mean']:+.4e}")

    # ---------------- baselines + Sec 3.6 diagnostic ------------------------
    extras = {}
    g_cmp = g_star if g_star > 0 else 1.0
    print(f"\n--- Baselines at position {i_star}, gamma={g_cmp} ---")
    for name, builder in [("naive_average", build_naive_average_layer),
                          ("duplicate", build_duplicate_layer)]:
        lay = builder(W.layers[i_star], W.layers[i_star + 1])
        cand = W.per_batch_losses(sel_pool, lay, i_star, g_cmp)
        st = paired_stats(base_sel, cand, n_boot=2000)
        extras[name] = dict(gamma=g_cmp, mean=float(st["mean"]), ci95=list(st["ci95"]))
        print(f"  {name:15s} dL = {st['mean']:+.6e}  CI95={st['ci95']}")
    try:
        extras["matching"] = matching_diagnostics(W.layers[i_star], W.layers[i_star + 1])
        m = extras["matching"]
        print(f"  matching: identity {m['identity_cost']:.4f} vs optimal "
              f"{m['optimal_cost']:.4f} ({m['reduction_pct']:.1f}% red., "
              f"{m['rematched']}/{m['units']} re-matched)")
    except Exception as e:
        print(f"  matching diagnostic skipped: {e}")

    # ---------------- persist ------------------------------------------------
    layer_path = args.layer_out or (args.out + ".layer.pt")
    for p in (args.out, layer_path):
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    torch.save(dict(state_dict=cache[i_star].state_dict(), position=i_star,
                    gamma=g_star, tau=args.tau, model=args.model), layer_path)

    json.dump(dict(model=args.model, revision=args.revision, dataset=args.dataset, dtype=args.dtype,
                   params_m=W.n_params() / 1e6, n_layer=W.n_layer,
                   position=i_star, gamma=g_star, tau=args.tau,
                   base_sel=float(base_sel.mean()),
                   sel_mean=best["mean"], sel_ci=best["ci95"], J=best["J"],
                   drift_unit=per_pos[i_star]["drift_unit"],
                   ceff=per_pos[i_star]["ceff"],
                   gate_config=cfg, screen=screen,
                   records={str(i): per_pos[i]["records"] for i in per_pos},
                   extras=extras, layer_file=layer_path,
                   block_size=args.block_size, batch=args.batch,
                   sel_batches=args.sel_batches, conf_batches=args.conf_batches,
                   max_tokens=args.max_tokens, eta=args.eta,
                   alt_rounds=args.alt_rounds, sinkhorn_iters=args.sinkhorn_iters),
              open(args.out, "w"), indent=2)
    print(f"\nSaved {args.out} and {layer_path}. Confirmation pool untouched. "
          f"Total {time.time()-t0:.0f}s.")


if __name__ == "__main__":
    main()
