"""Animated Figure 1: Sparse Motion Tokens by motion category, from real YAM eval video and the real SMT codes.
Pairs come from tools/smt_pool.py (same category, different episodes, chosen for token overlap). Each panel stacks the two
clips' token rows on one time axis. Tokens light up as the clips play; then the panel compares the rows: tokens in the
longest common subsequence (same codes, same order) are joined by lines and the shared sequence is spelled out.

  ~/traj_seg_vlm/.venv/bin/python tools/smt_motion_gif.py
  -> static/media/smt_motion.{gif,mp4,webm}, smt_motion_poster.png
"""
import json, subprocess, tempfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path.home() / "traj_seg_vlm"; SITE = Path(__file__).resolve().parents[1]; OUT = SITE / "static/media"
FPS, W, H = 10, 1280, 900
HOLD = 34                          # comparison phase, frames
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"; BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
f = lambda p, s: ImageFont.truetype(p, s)
BG, INK, DIM, FAINT, CARD, RULE = (245, 249, 255), (21, 35, 59), (68, 86, 111), (150, 163, 182), (255, 255, 255), (219, 231, 246)
PAL = [(42, 120, 214), (235, 104, 52), (27, 175, 122), (237, 161, 0), (232, 123, 164), (98, 80, 214), (227, 73, 72),
       (14, 154, 160), (0, 131, 0), (201, 133, 0)]
GREY = (214, 220, 230)
ORDER = ["Pick up", "Place", "Handover", "Bimanual grasp"]
LABEL = {"Pick up": "Single-arm pick up", "Place": "Single-arm place", "Handover": "Handover between arms",
         "Bimanual grasp": "Bimanual grasp"}


def clip_frames(ep, task, a, b, size):
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{a:.2f}", "-t", f"{b - a:.2f}", "-i", str(ROOT / f"proc/{task}/{ep}/video.mp4"),
                        "-vf", f"fps={FPS},crop=560:420:40:50,scale={size[0]}:{size[1]}", f"{d}/%04d.png"], check=True)
        return [Image.open(p).convert("RGB") for p in sorted(Path(d).glob("*.png"))]


def lcs(a, b):
    """Index pairs of a longest common subsequence of a and b."""
    n, m = len(a), len(b); L = [[0] * (m + 1) for _ in range(n + 1)]
    for x in range(n - 1, -1, -1):
        for y in range(m - 1, -1, -1):
            L[x][y] = L[x + 1][y + 1] + 1 if a[x] == b[y] else max(L[x + 1][y], L[x][y + 1])
    x = y = 0; out = []
    while x < n and y < m:
        if a[x] == b[y]: out.append((x, y)); x += 1; y += 1
        elif L[x + 1][y] >= L[x][y + 1]: x += 1
        else: y += 1
    return out


def dedup(seq):
    return [c for k, c in enumerate(seq) if k == 0 or c != seq[k - 1]]


def tint(c, a):
    return tuple(int(v * a + 255 * (1 - a)) for v in c)


def main():
    P = json.load(open(SITE / "tools/.smt_pool.json"))
    cells = []
    for k in ORDER:
        A, B = P["picks"][k]
        shared = sorted(set(A["codes"]) & set(B["codes"]))
        col = {c: PAL[i % len(PAL)] for i, c in enumerate(shared)}
        cells.append(dict(cat=k, shared=shared, col=col, clips=[
            dict(seg=s, frames=clip_frames(s["ep"], s["task"], s["start"], s["end"], (288, 216))) for s in (A, B)]))
    for cell in cells:
        A, B = (c["seg"]["codes"] for c in cell["clips"])
        cell["pairs"] = lcs(A, B)
        cell["seq"] = dedup([A[x] for x, _ in cell["pairs"]])
    T0 = max(len(c["frames"]) for cell in cells for c in cell["clips"]); T = T0 + HOLD
    frames = []
    for i in range(T):
        cmp_t = max(0, i - T0)                                                    # frames into the comparison phase
        im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
        d.text((40, 26), "Sparse Motion Tokens: recurring motion, recurring tokens", font=f(BOLD, 26), fill=INK)
        d.text((40, 62), "YAM robot · kinematics only · 2 tokens per second · 512-code vocabulary · lines join codes shared in the same order",
               font=f(FONT, 15), fill=DIM)
        for j, cell in enumerate(cells):
            x0, y0 = 40 + (j % 2) * 610, 98 + (j // 2) * 395
            d.rounded_rectangle((x0, y0, x0 + 590, y0 + 380), 12, fill=CARD, outline=RULE)
            d.text((x0 + 16, y0 + 12), LABEL[cell["cat"]], font=f(BOLD, 18), fill=INK)
            d.text((x0 + 574, y0 + 15), f"{len(cell['pairs'])} tokens in common order", font=f(FONT, 13), fill=DIM, anchor="ra")
            segs = [c["seg"] for c in cell["clips"]]
            nmax = max(len(sg["codes"]) for sg in segs); rx = x0 + 40; bw = min(30, 530 / max(nmax, 1))
            rows_y = (y0 + 242, y0 + 306); centers = []
            for m, c in enumerate(cell["clips"]):
                vx, vy = x0 + 47 + m * 260, y0 + 42
                fr = c["frames"][min(i, len(c["frames"]) - 1)].resize((236, 177))
                im.paste(fr, (vx, vy))
                d.rounded_rectangle((vx + 6, vy + 6, vx + 26, vy + 24), 4, fill=INK)
                d.text((vx + 16, vy + 15), "AB"[m], font=f(BOLD, 12), fill=(255, 255, 255), anchor="mm")
                seg = segs[m]; t_now = min(i / FPS, seg["end"] - seg["start"])
                d.text((vx + 2, vy + 181), seg["task"].replace("_", " ")[:36], font=f(FONT, 12), fill=FAINT)
                sy = rows_y[m]; matched = {p[m] for p in cell["pairs"]}
                d.text((x0 + 24, sy + 13), "AB"[m], font=f(BOLD, 13), fill=DIM, anchor="mm")
                cs = []
                for q, code in enumerate(seg["codes"]):
                    on = seg["times"][q] - seg["start"] <= t_now
                    base = cell["col"].get(code, GREY)
                    if cmp_t:                                                     # comparison: keep matched tokens, fade the rest
                        fill = base if q in matched else tint(base, 0.18)
                        txt = (255, 255, 255) if q in matched and code in cell["col"] else (175, 184, 198)
                    else:
                        fill = base if on else tint(base, 0.22)
                        txt = (255, 255, 255) if on and code in cell["col"] else ((60, 70, 85) if on else (190, 198, 210))
                    bx = rx + q * bw
                    d.rounded_rectangle((bx + 1, sy, bx + bw - 1, sy + 26), 4, fill=fill)
                    d.text((bx + bw / 2, sy + 13), str(code), font=f(MONO, 10 if bw < 26 else 11), fill=txt, anchor="mm")
                    cs.append(bx + bw / 2)
                centers.append(cs)
                if not cmp_t:
                    px = rx + min(t_now / 0.5, len(seg["codes"])) * bw
                    d.line((px, sy - 4, px, sy + 30), fill=INK, width=2)
            if cmp_t:                                                             # draw the alignment pair by pair
                shown = min(len(cell["pairs"]), 1 + cmp_t // 2)
                for (a, b) in cell["pairs"][:shown]:
                    code = segs[0]["codes"][a]
                    d.line((centers[0][a], rows_y[0] + 27, centers[1][b], rows_y[1] - 1), fill=cell["col"].get(code, DIM), width=2)
                if cmp_t >= 2 * len(cell["pairs"]):
                    sx, sy = x0 + 24, y0 + 348
                    d.text((sx, sy), "shared:", font=f(BOLD, 13), fill=INK, anchor="lm"); sx += 62
                    for n, code in enumerate(cell["seq"]):
                        tw = 34 if code >= 100 else 26
                        d.rounded_rectangle((sx, sy - 11, sx + tw, sy + 11), 4, fill=cell["col"].get(code, GREY))
                        d.text((sx + tw / 2, sy), str(code), font=f(MONO, 11), fill=(255, 255, 255), anchor="mm"); sx += tw
                        if n < len(cell["seq"]) - 1:
                            d.text((sx + 9, sy), "→", font=f(FONT, 13), fill=DIM, anchor="mm"); sx += 18
        frames.append(im)
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as d:
        for i, im in enumerate(frames): im.save(f"{d}/{i:04d}.png")
        frames[-1].save(OUT / "smt_motion_poster.png")              # poster = the comparison
        src = ["-framerate", str(FPS), "-i", f"{d}/%04d.png"]
        subprocess.run(["ffmpeg", "-v", "error", "-y", *src, "-c:v", "libx264", "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(OUT / "smt_motion.mp4")], check=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", *src, "-c:v", "libvpx-vp9", "-crf", "34", "-b:v", "0", str(OUT / "smt_motion.webm")], check=False)
        subprocess.run(["ffmpeg", "-v", "error", "-y", *src, "-vf", "scale=1120:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=160[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
                        str(OUT / "smt_motion.gif")], check=True)
    print("frames", len(frames), {p.name: round(p.stat().st_size / 1e6, 2) for p in OUT.glob("smt_motion*")})


if __name__ == "__main__":
    main()
