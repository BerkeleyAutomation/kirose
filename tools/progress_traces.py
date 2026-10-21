"""Progress traces for the title-figure player: for each robot episode (<= 70 s) in static/videos/segments.json, run the
KiRoSe progress head (frozen SMT encoder + signed-rate head, as in capabilities_fig.progress) inside every KiRoSe-predicted
segment, restarting at each segment start. CPU only.

  CUDA_VISIBLE_DEVICES="" ~/traj_seg_vlm/.venv/bin/python tools/progress_traces.py  -> static/videos/progress_traces.json
"""
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import capabilities_fig as C
H = C.H


def setup():
    net = C.DenseScorer(); net.load_state_dict(torch.load(C.ROOT / "ckpt_kin_signed2_yam.pt", map_location="cpu", weights_only=False)["model"]); net.eval()
    rng = np.random.default_rng(0)
    tr = [str(x) for x in np.load(C.ROOT / "data/chunks4v2/yam_train_physical.npz", allow_pickle=True)["ep_ids"]]
    samp = np.concatenate([H.load("yam_train", e)[0] for e in rng.choice(tr, 300, replace=False)])
    ck = torch.load(C.ROOT / "runs_progress_order/20260923/head_v2/v3_heads.pt", map_location="cpu", weights_only=False)
    h = C.RateHead(264); h.load_state_dict(ck["V2"]); h.eval()
    return net, torch.tensor(samp.mean(0)), torch.tensor(samp.std(0) + 1e-5), h, ck


def main():
    net, gm, gs, h, ck = setup()
    d = json.load(open(C.SITE / "static/videos/segments.json")); out = {}
    for g in d["groups"]:
        if g["key"] == "human":
            continue
        for r in g["runs"]:
            if r["duration"] > 70:
                continue
            split = next(s for s in ({"agibot": ["agibot_test", "agibot_ood"], "yam": ["yam_eval"]}[g["key"]])
                         if (C.ROOT / f"data/signed_gt2/{s}/{r['id']}.npz").exists())
            kin, kt = H.load(split, r["id"]); T, P = [], []
            for seg in r["kirose"]:
                a, b = seg["s"], min(seg["e"], float(kt[-1]))
                if b - a < 1.0:
                    continue
                t = np.arange(a + H.STEP, b + 1e-6, H.STEP)
                with torch.no_grad():
                    X = H.feats(net, kin, kt, a, t, gm, gs)
                    p = np.cumsum(h(torch.tensor((X - ck["mu"]) / ck["sd"], dtype=torch.float32)).numpy() * H.STEP)
                T += [round(a, 2)] + [round(float(x), 2) for x in t]; P += [0.0] + [round(float(x), 3) for x in p]
            out[r["id"]] = dict(t=T, p=P)
            print(g["key"], r["id"][:16], len(T), round(max(P), 2), round(min(P), 2), flush=True)
    json.dump(out, open(C.SITE / "static/videos/progress_traces.json", "w"), separators=(",", ":"))


if __name__ == "__main__":
    main()
