"""Pool of YAM eval segments by motion category with their Sparse Motion Tokens; Jaccard stats and pair selection.
  ~/.venv/bin/python tools/smt_pool.py  -> tools/.smt_pool.json"""
import json, glob, os, itertools, random
from pathlib import Path
import numpy as np
ROOT = Path.home() / "traj_seg_vlm"
CATS = {"Bimanual grasp": lambda d: "both arms" in d and not d.startswith("reset"),
        "Handover": lambda d: d.startswith("pass "),
        "Pick up": lambda d: d.startswith("pick up"),
        "Place": lambda d: d.startswith("place ")}
pool = []
for m in sorted(glob.glob(str(ROOT / "proc/*/*/meta.json"))):
    ep = os.path.basename(os.path.dirname(m)); task = os.path.basename(os.path.dirname(os.path.dirname(m)))
    cp = ROOT / f"kin_codes_signed/{ep}.json"
    if not cp.exists(): continue
    C = json.load(open(cp)); t = np.asarray(C["times"]); c = np.asarray(C["codes"])
    for s in json.load(open(m))["segments"]:
        d = s["description"].strip(); l = d.lower(); dur = s["end"] - s["start"]
        cat = next((k for k, f in CATS.items() if f(l)), None)
        if cat is None or not (2.5 <= dur <= 8.0): continue
        sel = (t >= s["start"]) & (t < s["end"])
        pool.append(dict(cat=cat, ep=ep, task=task, start=s["start"], end=s["end"], desc=d, codes=c[sel].tolist(), times=t[sel].tolist()))
jac = lambda a, b: len(set(a) & set(b)) / max(1, len(set(a) | set(b)))
rng = random.Random(0)
by = {k: [p for p in pool if p["cat"] == k] for k in CATS}
print({k: len(v) for k, v in by.items()})
same, diff = [], []
for _ in range(20000):
    a, b = rng.sample(pool, 2)
    if a["ep"] == b["ep"]: continue
    (same if a["cat"] == b["cat"] else diff).append(jac(a["codes"], b["codes"]))
print(f"Jaccard same-category {np.mean(same):.3f} (n={len(same)})  different-category {np.mean(diff):.3f} (n={len(diff)})")
# per-category same vs other
stats = {}
for k, v in by.items():
    s_ = [jac(a["codes"], b["codes"]) for a, b in itertools.combinations(v[:300], 2) if a["ep"] != b["ep"]]
    o_ = [jac(a["codes"], b["codes"]) for a in v[:200] for b in rng.sample([p for p in pool if p["cat"] != k], 20) if a["ep"] != b["ep"]]
    stats[k] = (float(np.mean(s_)), float(np.mean(o_))); print(k, f"same {np.mean(s_):.3f} other {np.mean(o_):.3f}")
# best pairs: different tasks preferred, high jaccard, 3-7 s
picks = {}
for k, v in by.items():
    best = None
    for a, b in itertools.combinations(v, 2):
        if a["ep"] == b["ep"]: continue
        if not (3 <= a["end"] - a["start"] <= 7 and 3 <= b["end"] - b["start"] <= 7): continue
        sc = jac(a["codes"], b["codes"]) + (0.15 if a["task"] != b["task"] else 0)
        if best is None or sc > best[0]: best = (sc, a, b)
    picks[k] = best[1:]; print(k, round(best[0], 3), best[1]["task"], best[1]["desc"], "|", best[2]["task"], best[2]["desc"])
json.dump(dict(picks=picks, stats=stats, same=float(np.mean(same)), diff=float(np.mean(diff)), n={k: len(v) for k, v in by.items()}),
          open(Path(__file__).parent / ".smt_pool.json", "w"), indent=1)
