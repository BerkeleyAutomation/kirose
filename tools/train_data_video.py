"""Animated training-data figure for the title section: what KiRoSe was trained on (YAM, ABC-130K train split).
Counters, a wall of the 140 tasks (one real arm-speed trace per task), real annotated timelines with their boundaries,
and the verb distribution of the YAM annotations. All numbers come from the training split.

  ~/traj_seg_vlm/.venv/bin/python tools/train_data_video.py   -> static/media/train_data.{mp4,webm}, train_data_poster.png
"""
import json, subprocess, tempfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path.home() / "traj_seg_vlm"; SITE = Path(__file__).resolve().parents[1]; OUT = SITE / "static/media"
FPS, W, H, S = 10, 1280, 720, 2                      # drawn at 2x, downsampled for crisp lines
BUILD, TOTAL = 60, 170                               # frames: build-up, then highlight cycle
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"; BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
f = lambda p, s: ImageFont.truetype(p, s * S)
BG, INK, DIM, FAINT, CARD, RULE = (245, 249, 255), (21, 35, 59), (68, 86, 111), (116, 136, 163), (255, 255, 255), (219, 231, 246)
ACC, ORG = (37, 99, 235), (235, 104, 52)
PAL = [(42, 120, 214), (235, 104, 52), (27, 175, 122), (237, 161, 0), (232, 123, 164), (98, 80, 214), (227, 73, 72),
       (14, 154, 160), (0, 131, 0), (201, 133, 0)]
ease = lambda x: 1 - (1 - min(max(x, 0), 1)) ** 3


def load():
    d = np.load(ROOT / "data/chunks4v2/yam_train_physical.npz", allow_pickle=True)
    ids, task, dur, gt = [str(x) for x in d["ep_ids"]], [str(x) for x in d["ep_task"]], d["ep_dur"], d["ep_gt"]
    rng = np.random.default_rng(3); tasks = sorted(set(task)); rng.shuffle(tasks)
    traces = {}
    for tk in tasks:                                  # one episode per task: arm speeds, smoothed, 60 points
        i = next(j for j in rng.permutation(len(ids)) if task[j] == tk)
        z = np.load(ROOT / f"cache_signed2_yam/{ids[i]}.npz")
        tr = []
        for arm in ("left", "right"):
            v = np.linalg.norm(z[f"feat_{arm}"][:, :3], axis=1); t = z[f"t_{arm}"]
            g = np.linspace(t[0], t[-1], 60); tr.append(np.interp(g, t, np.convolve(v, np.ones(61) / 61, "same")))
        traces[tk] = np.array(tr)
    tl = [i for i in rng.permutation(len(ids)) if 120 < dur[i] < 260][:9]
    verbs = json.load(open(ROOT / "results/raw_annotation_verbs.json"))["YAM"]["first_word_counts"]
    nb = sum(len(g) for g in gt)
    stats = [("Episodes", len(ids)), ("Hours", round(float(dur.sum()) / 3600)), ("Tasks", len(tasks)),
             ("Timed events", int(nb + len(ids))), ("Distinct verbs", 83)]
    return tasks, traces, [(task[i], float(dur[i]), np.asarray(gt[i], float)) for i in tl], verbs, stats


def text(d, xy, s, font, fill, anchor="la"): d.text((xy[0] * S, xy[1] * S), s, font=font, fill=fill, anchor=anchor)
def rect(d, x0, y0, x1, y1, **k): d.rounded_rectangle((x0 * S, y0 * S, x1 * S, y1 * S), **{"radius": 6 * S, **k})
def name(tk): return tk.replace("_", " ").capitalize()


def frame(i, tasks, traces, tls, verbs, stats):
    im = Image.new("RGB", (W * S, H * S), BG); d = ImageDraw.Draw(im); p = ease(i / BUILD)
    text(d, (32, 26), "What KiRoSe is trained on", f(BOLD, 26), INK)
    text(d, (32, 62), "YAM bimanual robot, ABC-130K training split: end-effector kinematics + subtask descriptions with start and end times",
         f(FONT, 14), DIM)
    # counters
    cw = (W - 64 - 4 * 14) / 5
    for k, (lab, v) in enumerate(stats):
        x = 32 + k * (cw + 14); rect(d, x, 96, x + cw, 166, fill=CARD, outline=RULE, width=S)
        text(d, (x + 16, 104), f"{round(v * p):,}", f(BOLD, 30), ACC); text(d, (x + 16, 144), lab, f(FONT, 13), DIM)
    # task wall
    ax, ay, aw = 32, 196, 600
    text(d, (ax, ay), "140 household tasks", f(BOLD, 16), INK)
    hi = tasks[((i - BUILD) // 11) % len(tasks)] if i >= BUILD else None
    text(d, (ax + aw, ay + 2), ("e.g. " + name(hi)) if hi else "one real arm-speed trace per task", f(FONT, 13), ACC if hi else FAINT, "ra")
    cols, rows, g = 14, 10, 5; tw, th = (aw - (cols - 1) * g) / cols, (690 - ay - 28 - (rows - 1) * g) / rows
    shown = int(len(tasks) * ease(i / (BUILD * 0.8)))
    for k, tk in enumerate(tasks[:shown]):
        x, y = ax + (k % cols) * (tw + g), ay + 28 + (k // cols) * (th + g)
        on = tk == hi
        rect(d, x, y, x + tw, y + th, radius=4 * S, fill=(232, 240, 253) if on else CARD, outline=ACC if on else RULE, width=(2 if on else 1) * S)
        tr = traces[tk]; m = tr.max() + 1e-6
        for a, c in ((0, INK), (1, ORG)):
            pts = [((x + 3 + j / 59 * (tw - 6)) * S, (y + th - 4 - tr[a, j] / m * (th - 10)) * S) for j in range(60)]
            d.line(pts, fill=c, width=S)
    # timelines
    bx, by, bw = 672, 196, W - 32 - 672
    text(d, (bx, by), "Dense timed annotations", f(BOLD, 16), INK)
    text(d, (bx + bw, by + 2), "8.8 boundaries per minute", f(FONT, 13), FAINT, "ra")
    T, lx = 260.0, bx + 6; sweep = T * ease(i / BUILD)
    for r, (tk, du, g_) in enumerate(tls):
        y = by + 30 + r * 25; edges = np.r_[0, g_, du]
        for s in range(len(edges) - 1):
            a, b = edges[s], min(edges[s + 1], sweep)
            if b <= a: break
            c = PAL[s % len(PAL)]
            d.rectangle(((lx + a / T * (bw - 6)) * S, y * S, (lx + b / T * (bw - 6)) * S, (y + 18) * S), fill=c, outline=CARD, width=S)
    for t in range(0, 261, 60):
        x = lx + t / T * (bw - 6); text(d, (x, by + 30 + 9 * 25 + 2), f"{t}s", f(FONT, 11), FAINT, "ma")
    # verbs
    vx, vy = 672, 490
    text(d, (vx, vy), "Most common verbs", f(BOLD, 16), INK)
    text(d, (vx + bw, vy + 2), "share of annotated events", f(FONT, 13), FAINT, "ra")
    tot = sum(verbs.values()); top = list(verbs.items())[:10]; colw = (bw - 24) / 2
    for k, (v, n) in enumerate(top):
        x, y = vx + (k // 5) * (colw + 24), vy + 32 + (k % 5) * 34
        frac = n / tot; bar = (colw - 112) * frac / (top[0][1] / tot) * ease((i - k * 2) / BUILD)
        text(d, (x, y + 2), v, f(FONT, 13), INK)
        d.rounded_rectangle(((x + 72) * S, (y + 2) * S, (x + 72 + max(bar, 1)) * S, (y + 18) * S), radius=3 * S, fill=ACC)
        text(d, (x + 76 + bar, y + 3), f"{100 * frac * min(p * 1.2, 1):.0f}%", f(FONT, 12), DIM)
    return im.resize((W, H), Image.LANCZOS)


def main():
    OUT.mkdir(parents=True, exist_ok=True); data = load()
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(TOTAL):
            im = frame(i, *data); im.save(f"{tmp}/{i:04d}.png")
            if i == BUILD + 3: im.save(OUT / "train_data_poster.png")
        src = ["-framerate", str(FPS), "-i", f"{tmp}/%04d.png"]
        subprocess.run(["ffmpeg", "-v", "error", "-y", *src, "-c:v", "libx264", "-crf", "22", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(OUT / "train_data.mp4")], check=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", *src, "-c:v", "libvpx-vp9", "-crf", "34", "-b:v", "0", str(OUT / "train_data.webm")], check=True)
    print("ok", data[4])


if __name__ == "__main__":
    main()
