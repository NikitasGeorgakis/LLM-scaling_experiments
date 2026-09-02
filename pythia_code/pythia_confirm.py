"""
STAGE 2 -- CONFIRMATION (paper eq. 3.48, second half), full-fidelity version.

Loads the EXACT layer parameters saved by selection (no rebuild), prints the
pre-registered decision rule BEFORE computing anything, then runs the single
out-of-sample test on the confirmation pool.

Retention (eq. 3.45 with the materiality margin of eq. 3.47):
    keep gamma* > 0  <=>  mean paired diff < -delta  AND  bootstrap CI95 upper < 0
Otherwise gamma* = 0 and the original pretrained model is recovered exactly.

Usage:
    python pythia_confirm.py --selection results/sel_1b.json --out results/conf_1b.json
"""
import argparse, copy, json, os, time
import torch

from pythia_wrapper import PythiaWrapper
from pythia_data import load_token_stream, make_pools
from pythia_stats import paired_stats, print_rule, verdict, DELTA_MATERIAL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", required=True)
    ap.add_argument("--out", default="results/confirmation.json")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--exploratory", default="",
                    help="comma-separated extra gammas, reported POST-DECISION only")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    sel = json.load(open(args.selection))
    print(f"Loaded selection: {sel['model']}  position={sel['position']}  "
          f"gamma={sel['gamma']}  tau={sel['tau']}  (J={sel['J']:+.3e})\n")

    if sel["gamma"] == 0.0:
        print("Selection already returned gamma* = 0: no feasible gamma > 0 beat "
              "doing nothing on the selection set. Original model retained "
              "exactly; nothing to confirm.")
        json.dump(dict(model=sel["model"], revision=sel.get("revision"), n_layer=sel.get("n_layer"),
                       params_m=sel.get("params_m"), position=sel["position"], gamma=0.0,
                       retained=False, note="gamma*=0 at selection"),
                  open(args.out, "w"), indent=2)
        return

    print_rule()
    print()

    torch.manual_seed(2026)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if sel.get("dtype", "float32") == "float32" else torch.bfloat16
    W = PythiaWrapper(sel["model"], dtype=dtype, device=dev, cache_dir=args.cache_dir, revision=sel.get("revision"))
    print("[model]", W.describe())

    # exact saved parameters -- no rebuild
    saved = torch.load(sel["layer_file"], map_location=dev)
    assert saved["position"] == sel["position"] and saved["model"] == sel["model"]
    layer = copy.deepcopy(W.layers[sel["position"]])
    layer.load_state_dict(saved["state_dict"])
    layer.eval()

    stream = load_token_stream(W.tokenizer, sel["dataset"], sel["max_tokens"],
                               args.cache_dir)
    _, conf_pool = make_pools(stream, sel["block_size"], sel["batch"],
                              sel["sel_batches"], sel["conf_batches"])

    t0 = time.time()
    i, g = sel["position"], sel["gamma"]
    base = W.per_batch_losses(conf_pool)
    cand = W.per_batch_losses(conf_pool, layer, i, g)
    st = paired_stats(base, cand, n_boot=10000)

    print(f"\n[base] CONFIRMATION loss = {base.mean():.6f} nats/token")
    print(f"[cand] CONFIRMATION loss = {cand.mean():.6f} nats/token\n")
    print("CONFIRMATION RESULT (single pre-registered test)")
    print(f"  n batches   = {st['n']}")
    print(f"  mean dL     = {st['mean']:+.7e}")
    print(f"  SE          = {st['se']:.3e}")
    print(f"  t           = {st['t']:+.3f}")
    print(f"  p (1-sided) = {st['p_one_sided']:.5f}")
    print(f"  CI95        = [{st['ci95'][0]:+.4e}, {st['ci95'][1]:+.4e}]")

    ok, material, signif = verdict(st)
    print(f"\n  (i)  materiality   mean < -{DELTA_MATERIAL} : {'PASS' if material else 'FAIL'}")
    print(f"  (ii) significance  CI95 upper < 0   : {'PASS' if signif else 'FAIL'}")
    if ok:
        print(f"\n  VERDICT: SUBSTANTIVE IMPROVEMENT -- retain layer at gamma* = {g}")
    elif signif:
        print("\n  VERDICT: STATISTICALLY DETECTABLE BUT NOT MATERIAL -- gamma* = 0")
    else:
        print("\n  VERDICT: NOT CONFIRMED -- gamma* = 0 (original model recovered exactly)")

    # descriptive (not part of the decision rule): KL and drift at gamma* on
    # a few confirmation batches, and latency overhead
    kl = []
    for b in conf_pool[:8]:
        bl, _ = W.logits_and_loss(b)
        with W.inserted(layer, i, g):
            cl, _ = W.logits_and_loss(b)
        kl.append(W.kl_to_base(bl, cl))
    drift1 = W.measure_drift_unit(layer, i, conf_pool[:4])
    t_base, t_ext = W.measure_latency(layer, i, conf_pool[:2])
    print(f"\n  descriptive at gamma*={g}:  D_KL^out = {sum(kl)/len(kl):.3e}  "
          f"D_rep = {(g**2)*drift1:.3e}  latency overhead = "
          f"{max(0.0,(t_ext-t_base)/t_base)*100:.2f}%")

    out = dict(model=sel["model"], revision=sel.get("revision"), dataset=sel["dataset"], params_m=sel["params_m"],
               n_layer=sel["n_layer"], position=i, gamma=g, tau=sel["tau"],
               base_conf=float(base.mean()), cand_conf=float(cand.mean()),
               n=int(st["n"]), mean=float(st["mean"]), se=float(st["se"]),
               t=float(st["t"]), p_one_sided=float(st["p_one_sided"]),
               ci95=list(st["ci95"]), delta_material=DELTA_MATERIAL,
               material=bool(material), significant=bool(signif),
               retained=bool(ok),
               kl_at_gstar=float(sum(kl)/len(kl)),
               drep_at_gstar=float((g**2)*drift1),
               latency_overhead=float(max(0.0, (t_ext - t_base) / t_base)))

    if args.exploratory:
        print("\n--- POST-DECISION EXPLORATORY (not pre-registered) ---")
        out["exploratory"] = []
        for gx in [float(x) for x in args.exploratory.split(",")]:
            c = W.per_batch_losses(conf_pool, layer, i, gx)
            s = paired_stats(base, c, n_boot=10000)
            print(f"  gamma={gx:<6.3f} dL={s['mean']:+.4e}  "
                  f"CI95=[{s['ci95'][0]:+.4e},{s['ci95'][1]:+.4e}]")
            out["exploratory"].append(dict(gamma=gx, mean=float(s["mean"]),
                                           ci95=list(s["ci95"])))

    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nSaved {args.out}. Total {time.time()-t0:.0f}s.")


if __name__ == "__main__":
    main()
