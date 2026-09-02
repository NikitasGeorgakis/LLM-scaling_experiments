"""Prints 'POS GAMMA' for the pre-declared loss-only secondary candidate
(stability-feasible, minimum mean dL), or 'NONE'. Rule fixed before any
revision data is seen."""
import json, sys
s = json.load(open(sys.argv[1]))
cands = [(r["mean"], r["gamma"], int(p)) for p, recs in s["records"].items()
         for r in recs if r["gamma"] > 0 and r["mean"] <= 0
         and r["kl"] <= 0.05 and r["drift"] <= 0.05]
print("NONE" if not cands else f"{min(cands)[2]} {min(cands)[1]}")
