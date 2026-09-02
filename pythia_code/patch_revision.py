import subprocess, sys
ok = True
def patch(path, subs):
    global ok
    s = open(path).read()
    for old, new in subs:
        if old not in s:
            print(f"FAIL {path}: pattern not found: {old[:60]}...")
            ok = False
            return
        s = s.replace(old, new, 1)
    open(path, "w").write(s)
    print(f"patched {path}")

patch("pythia_wrapper.py", [
 ('def __init__(self, model_name="EleutherAI/pythia-1b", dtype=torch.float32,\n                 device="cuda", cache_dir=None):',
  'def __init__(self, model_name="EleutherAI/pythia-1b", dtype=torch.float32,\n                 device="cuda", cache_dir=None, revision=None):'),
 ('self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)',
  'self.revision = revision\n        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, revision=revision)'),
 ('AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype,\n                                                     cache_dir=cache_dir)',
  'AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype,\n                                                     cache_dir=cache_dir, revision=revision)'),
 ('return (f"{self.model_name}: n_layer=',
  'return (f"{self.model_name}@{getattr(self,\'revision\',None) or \'main\'}: n_layer='),
])
patch("pythia_select.py", [
 ('ap.add_argument("--cache-dir", default=None)',
  'ap.add_argument("--cache-dir", default=None)\n    ap.add_argument("--revision", default=None, help="HF revision, e.g. step8000")'),
 ('W = PythiaWrapper(args.model, dtype=dtype, device=dev, cache_dir=args.cache_dir)',
  'W = PythiaWrapper(args.model, dtype=dtype, device=dev, cache_dir=args.cache_dir, revision=args.revision)'),
 ('json.dump(dict(model=args.model, dataset=args.dataset, dtype=args.dtype,',
  'json.dump(dict(model=args.model, revision=args.revision, dataset=args.dataset, dtype=args.dtype,'),
])
patch("pythia_confirm.py", [
 ('W = PythiaWrapper(sel["model"], dtype=dtype, device=dev, cache_dir=args.cache_dir)',
  'W = PythiaWrapper(sel["model"], dtype=dtype, device=dev, cache_dir=args.cache_dir, revision=sel.get("revision"))'),
 ('json.dump(dict(model=sel["model"], position=sel["position"], gamma=0.0,\n                       retained=False, note="gamma*=0 at selection"),',
  'json.dump(dict(model=sel["model"], revision=sel.get("revision"), n_layer=sel.get("n_layer"),\n                       params_m=sel.get("params_m"), position=sel["position"], gamma=0.0,\n                       retained=False, note="gamma*=0 at selection"),'),
 ('out = dict(model=sel["model"], dataset=sel["dataset"], params_m=sel["params_m"],',
  'out = dict(model=sel["model"], revision=sel.get("revision"), dataset=sel["dataset"], params_m=sel["params_m"],'),
])
patch("confirm_secondary.py", [
 ('W = PythiaWrapper(sel["model"], dtype=torch.float32, device=dev)',
  'W = PythiaWrapper(sel["model"], dtype=torch.float32, device=dev, revision=sel.get("revision"))'),
 ('json.dump(dict(model=sel["model"], position=a.position, gamma=a.gamma, secondary=True,',
  'json.dump(dict(model=sel["model"], revision=sel.get("revision"), position=a.position, gamma=a.gamma, secondary=True,'),
])
if ok:
    r = subprocess.run([sys.executable, "-m", "py_compile", "pythia_wrapper.py",
                        "pythia_select.py", "pythia_confirm.py", "confirm_secondary.py"])
    print("ALL PATCHED, syntax OK" if r.returncode == 0 else "SYNTAX ERROR")
else:
    print("SOME PATCHES FAILED - send me this output, nothing was broken")
