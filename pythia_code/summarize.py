"""Trend table across sizes AND training stages (revisions)."""
import glob, json
rows = []
for f in sorted(glob.glob("results/conf_*.json")):
    c = json.load(open(f))
    label = c["model"] + ("@" + c["revision"] if c.get("revision") else "")
    if "confsec" in f:
        label += " [SEC]"
    if "mean" not in c:
        rows.append((label, c.get("n_layer","?"), c.get("position"), c.get("gamma"),
                     None, None, None, c.get("note","gamma*=0")))
        continue
    v = ("SUBSTANTIVE" if c.get("retained") else
         "detectable, not material" if c.get("significant") else "not confirmed")
    rows.append((label, c.get("n_layer","?"), c["position"], c["gamma"],
                 c["mean"], c.get("ci95"), c.get("p_one_sided", c.get("p")), v))
print(f"{'model':38s} {'L':>3} {'pos':>3} {'gamma':>7} {'conf dL':>11} "
      f"{'CI95':>26} {'p':>7}  verdict")
for label, L, pos, g, mean, ci, p, v in rows:
    if mean is None:
        print(f"{label:38s} {str(L):>3} {str(pos):>3} {str(g):>7} {'-':>11} {'-':>26} {'-':>7}  {v}")
    else:
        print(f"{label:38s} {str(L):>3} {pos:>3} {g:>7.4g} {mean:>+11.3e} "
              f"[{ci[0]:+.2e},{ci[1]:+.2e}] {p:>7.4f}  {v}")
