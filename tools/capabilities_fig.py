"""Capabilities figure for the KiRoSe release: one AgiBot G1 episode (zero-shot), kinematics only.
Rows: reference video frames (not a model input) | kinematics | Sparse Motion Tokens | annotation vs KiRoSe segments with
descriptions | progress head on one annotated subtask, as recorded and reversed in place. Runs on CPU.

  ~/traj_seg_vlm/.venv/bin/python tools/capabilities_fig.py --ep 655509 --unit 1   -> static/images/capabilities.{png,pdf}, capabilities.json
"""
import argparse, json, subprocess, sys, textwrap
from pathlib import Path
import numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path.home() / "traj_seg_vlm"; SITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments/progress_order_20260923"))
import head_v2 as H, diag_progress as DP
H.dev = DP.dev = "cpu"
from model_event_signed import DenseScorer

INK, MUTED, GRID = "#16181b", "#80848b", "#e6e6e1"
KIR, GT, FWD, REV = "#2a78d6", "#4c5057", "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})


class RateHead(nn.Module):       # head_v3.py V2: signed progress rate, progress = running sum
    def __init__(self, d): super().__init__(); self.net = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 1))
    def forward(self, x): return self.net(x).squeeze(-1)


def progress(ep, split, a, b):
    net = DenseScorer(); net.load_state_dict(torch.load(ROOT / "ckpt_kin_signed2_yam.pt", map_location="cpu", weights_only=False)["model"]); net.eval()
    rng = np.random.default_rng(0)                    # same draw as head_v3.py -> identical YAM-train standardization
    tr = [str(x) for x in np.load(ROOT / "data/chunks4v2/yam_train_physical.npz", allow_pickle=True)["ep_ids"]]
    samp = np.concatenate([H.load("yam_train", e)[0] for e in rng.choice(tr, 300, replace=False)])
    gm, gs = torch.tensor(samp.mean(0)), torch.tensor(samp.std(0) + 1e-5)
    ck = torch.load(ROOT / "runs_progress_order/20260923/head_v2/v3_heads.pt", map_location="cpu", weights_only=False)
    h = RateHead(264); h.load_state_dict(ck["V2"]); h.eval()
    kin, kt = H.load(split, ep); t = np.arange(a + H.STEP, b + 1e-6, H.STEP)
    out = {}
    for name, k in (("forward", kin), ("reversed", H.rev_kin(kin, kt, a, b))):
        X = H.feats(net, k, kt, a, t, gm, gs)
        with torch.no_grad():
            out[name] = np.cumsum(h(torch.tensor((X - ck["mu"]) / ck["sd"], dtype=torch.float32)).numpy() * H.STEP)
    return t - a, out, kin, kt


def frames(ep, times, out):
    out.mkdir(parents=True, exist_ok=True); fs = []
    for i, t in enumerate(times):
        p = out / f"{ep}_{i}.jpg"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.2f}", "-i", str(ROOT / f"data/agibot_video/{ep}.mp4"), "-frames:v", "1", "-vf", "scale=320:-2", str(p)], check=True)
        fs.append(plt.imread(p))
    return fs


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ep", default="655509"); ap.add_argument("--split", default="test"); ap.add_argument("--unit", type=int, default=1)
    a = ap.parse_args()
    rec = next(r for r in map(json.loads, open(ROOT / f"runs_signed_reader/20260923_trainonly/preds_agibot_{a.split}.jsonl")) if r["episode_id"] == a.ep)
    codes = json.load(open(ROOT / f"kin_codes_signed_trainonly/{a.ep}.json"))
    gt = sorted(rec["gt_segments"], key=lambda s: s["start"]); pred = sorted(rec["pred_segments"], key=lambda s: s["start"]); T = rec["duration_s"]
    u = gt[a.unit]; tt, prog, kin, kt = progress(a.ep, f"agibot_{a.split}", u["start"], u["end"])

    fig = plt.figure(figsize=(11, 9.6))
    gsp = fig.add_gridspec(5, 6, height_ratios=[1.15, 1.0, 0.32, 1.9, 1.55], hspace=0.75, wspace=0.08)
    # 1 reference frames
    ft = np.linspace(2, T - 3, 6)
    for i, (img, t) in enumerate(zip(frames(a.ep, ft, SITE / "tools/.frames"), ft)):
        ax = fig.add_subplot(gsp[0, i]); ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
        ax.set_title(f"{t:.0f} s", fontsize=8, color=MUTED, pad=2)
    fig.text(0.125, 0.905, f'AgiBot G1 · "{rec["instruction"]}" · never seen in training · video shown for reference only, KiRoSe reads kinematics',
             fontsize=9, color=MUTED)
    # 2 kinematics
    ax = fig.add_subplot(gsp[1, :])
    for arm, off, c in (("left", 0, INK), ("right", 7, REV)):
        ax.plot(kt, np.linalg.norm(kin[:, off:off + 3], axis=1), color=c, lw=1, label=f"{arm} arm speed")
    ax.set_xlim(0, T); ax.set_ylabel("speed"); ax.set_xticklabels([]); ax.grid(axis="x", color=GRID, lw=0.6)
    ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=2)
    ax.set_title("① Input: end-effector kinematics (14 channels; arm speeds shown)", loc="left", fontsize=10, color=INK, fontweight="bold")
    # 3 tokens
    ax = fig.add_subplot(gsp[2, :]); cm = plt.get_cmap("tab20")
    ct = np.asarray(codes["times"], float); cc = np.asarray(codes["codes"], int); w = np.median(np.diff(ct)) if len(ct) > 1 else 0.5
    for t, c in zip(ct, cc): ax.add_patch(Rectangle((t - w / 2, 0), w * 0.92, 1, color=cm(c % 20), lw=0))
    ax.set_xlim(0, T); ax.set_ylim(0, 1); ax.set_yticks([]); ax.set_xticklabels([]); [s.set_visible(False) for s in ax.spines.values()]
    ax.set_title(f"② Sparse Motion Tokens: 2 per second, {len(cc)} tokens for {T:.0f} s (colour = code id)", loc="left", fontsize=10, color=INK, fontweight="bold")
    # 4 segments
    ax = fig.add_subplot(gsp[3, :])
    for row, segs, c, lab in ((1, gt, GT, "Annotation"), (0, pred, KIR, "KiRoSe")):
        for i, s in enumerate(segs):
            ax.add_patch(Rectangle((s["start"], row + 0.08), s["end"] - s["start"], 0.84, facecolor=c, alpha=0.18 if i % 2 else 0.32, edgecolor="white", lw=1.5))
            d = s.get("description", "")
            if s["end"] - s["start"] > 2.5 and d:
                ax.text((s["start"] + s["end"]) / 2, row + 0.5, textwrap.fill(d.rstrip("."), 24 if row else 18), ha="center", va="center", fontsize=6.2 if row else 6.8, color=INK)
        ax.text(-0.4, row + 0.5, lab, ha="right", va="center", fontsize=9, color=c, fontweight="bold")
    ax.add_patch(Rectangle((u["start"], 1.02), u["end"] - u["start"], 0.96, fill=False, edgecolor=REV, lw=1.8))
    ax.set_xlim(0, T); ax.set_ylim(0, 2); ax.set_yticks([]); ax.set_xlabel("time (s)")
    ax.set_title("③ Output: subtask boundaries and action descriptions (Qwen3.5-9B reader)", loc="left", fontsize=10, color=INK, fontweight="bold")
    # 5 progress
    ax = fig.add_subplot(gsp[4, 1:5])
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.plot(tt, prog["forward"], color=FWD, lw=2.2, label="as recorded")
    ax.plot(tt, prog["reversed"], color=REV, lw=2.2, label="same subtask played in reverse")
    ax.set_xlabel("time within subtask (s)"); ax.set_ylabel("predicted progress"); ax.grid(color=GRID, lw=0.6)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    ax.set_title("④ Output: dense progress on the boxed subtask (68k-parameter head on the same tokenizer)",
                 loc="left", fontsize=10, color=INK, fontweight="bold")
    for ext in ("png", "pdf"):
        fig.savefig(SITE / f"static/images/capabilities.{ext}", dpi=170, bbox_inches="tight", facecolor="white")
    json.dump(dict(episode=a.ep, instruction=rec["instruction"], unit=u, forward_end=float(prog["forward"][-1]), reversed_end=float(prog["reversed"][-1]),
                   n_tokens=int(len(cc)), duration=T), open(SITE / "static/images/capabilities.json", "w"), indent=1)
    print("ok", a.ep, u, prog["forward"][-1], prog["reversed"][-1])


if __name__ == "__main__":
    main()
