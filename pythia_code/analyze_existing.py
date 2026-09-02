import json, glob
print("=== small-gamma behaviour from EXISTING selection records (no new compute) ===")
for f in sorted(glob.glob("results/sel_*_pile.json")):
    s = json.load(open(f))
    print(f"\n{s['model']}  L={s['n_layer']}  {s['params_m']:.0f}M  base={s['base_sel']:.4f}")
    for pos, recs in s["records"].items():
        neg = [r for r in recs if r["gamma"] > 0 and r["mean"] < 0]
        best = min((r for r in recs if r["gamma"] > 0), key=lambda r: r["mean"])
        print(f"  pos {pos}: best dL={best['mean']:+.3e} @ g={best['gamma']}"
              f"  CI95=[{best['ci95'][0]:+.2e},{best['ci95'][1]:+.2e}]"
              f"  KL={best['kl']:.1e}  neg-dL gammas={[r['gamma'] for r in neg]}")
    # what a loss-only gate (stability constraints kept, C_eff dropped) would pick
    cands = [(r["mean"], r["gamma"], int(p)) for p, recs in s["records"].items()
             for r in recs if r["mean"] <= 0 and r["kl"] <= 0.05 and r["drift"] <= 0.05]
    dl, g, p = min(cands)
    print(f"  -> loss-only secondary pick: pos={p} gamma={g}  dL={dl:+.3e}")
