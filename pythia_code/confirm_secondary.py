"""SECONDARY, pre-declared single confirmation of the loss-only exploratory
candidate. Primary result (gamma*=0 under the compute-parity objective)
stands regardless; this is explicitly labelled exploratory in the paper."""
import argparse, copy, json, torch
from pythia_wrapper import PythiaWrapper
from pythia_data import load_token_stream, make_pools
from pythia_stats import paired_stats, print_rule, verdict, DELTA_MATERIAL

ap = argparse.ArgumentParser()
ap.add_argument("--selection", required=True)
ap.add_argument("--position", type=int, required=True)
ap.add_argument("--gamma", type=float, required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

sel = json.load(open(a.selection))
print(f"SECONDARY confirmation: {sel['model']}  pos={a.position}  gamma={a.gamma}")
print("Declared ONCE, on the untouched confirmation pool. Rule first:\n")
print_rule(); print()

torch.manual_seed(2026)
dev = "cuda" if torch.cuda.is_available() else "cpu"
W = PythiaWrapper(sel["model"], dtype=torch.float32, device=dev, revision=sel.get("revision"))
saved = torch.load(sel["layer_file"], map_location=dev)
layer = copy.deepcopy(W.layers[a.position])
if saved["position"] == a.position:
    layer.load_state_dict(saved["state_dict"]); print("[layer] loaded saved state (bit-identical to selection)")
else:
    from pythia_barycentric import build_barycentric_layer
    layer, _ = build_barycentric_layer(W.layers[a.position], W.layers[a.position+1],
        tau=sel["tau"], eta=sel["eta"], n_alt_rounds=sel["alt_rounds"],
        sinkhorn_iters=sel["sinkhorn_iters"])
    print("[layer] rebuilt deterministically for this position")
layer.eval()

stream = load_token_stream(W.tokenizer, sel["dataset"], sel["max_tokens"])
_, conf = make_pools(stream, sel["block_size"], sel["batch"],
                     sel["sel_batches"], sel["conf_batches"])
base = W.per_batch_losses(conf)
cand = W.per_batch_losses(conf, layer, a.position, a.gamma)
st = paired_stats(base, cand, n_boot=10000)
ok, material, signif = verdict(st)
print(f"\nmean dL = {st['mean']:+.4e}   CI95=[{st['ci95'][0]:+.4e},{st['ci95'][1]:+.4e}]   p={st['p_one_sided']:.4f}")
print(f"materiality (<-{DELTA_MATERIAL}): {'PASS' if material else 'FAIL'}   significance (CI<0): {'PASS' if signif else 'FAIL'}")
print("VERDICT:", "SUBSTANTIVE" if ok else ("DETECTABLE, NOT MATERIAL" if signif else "NOT CONFIRMED"))
json.dump(dict(model=sel["model"], revision=sel.get("revision"), position=a.position, gamma=a.gamma, secondary=True,
               mean=float(st["mean"]), ci95=list(st["ci95"]), p=float(st["p_one_sided"]),
               material=bool(material), significant=bool(signif), retained=bool(ok)),
          open(a.out, "w"), indent=2)
print(f"Saved {a.out}")
