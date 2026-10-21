"""Animated Figure 1: Sparse Motion Tokens by motion category, from real YAM eval video and the real SMT codes.
Pairs come from tools/smt_pool.py (same category, different episodes, chosen for token overlap). Codes shared by the
two clips of a category are coloured; codes that are not shared are grey. Tokens light up as the clip plays.

  ~/traj_seg_vlm/.venv/bin/python tools/smt_motion_gif.py
  -> static/media/smt_motion.{gif,mp4,webm}, smt_motion_poster.png
"""
import json, subprocess, tempfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path.home() / "traj_seg_vlm"; SITE = Path(__file__).resolve().parents[1]; OUT = SITE / "static/media"
FPS, W, H = 10, 1280, 800
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


def main():
    P = json.load(open(SITE / "tools/.smt_pool.json"))
    cells = []
    for k in ORDER:
        A, B = P["picks"][k]
        shared = sorted(set(A["codes"]) & set(B["codes"]))
        col = {c: PAL[i % len(PAL)] for i, c in enumerate(shared)}
        cells.append(dict(cat=k, shared=shared, col=col, clips=[
            dict(seg=s, frames=clip_frames(s["ep"], s["task"], s["start"], s["end"], (288, 216))) for s in (A, B)]))
    T = max(len(c["frames"]) for cell in cells for c in cell["clips"]) + FPS          # hold 1 s at the end
    frames = []
    for i in range(T):
        im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
        d.text((40, 26), "Sparse Motion Tokens: recurring motion, recurring tokens", font=f(BOLD, 26), fill=INK)
        d.text((40, 62), "YAM robot · kinematics only · 2 tokens per second · 512-code vocabulary · coloured = code shared by both clips",
               font=f(FONT, 15), fill=DIM)
        for j, cell in enumerate(cells):
            x0, y0 = 40 + (j % 2) * 610, 100 + (j // 2) * 345
            d.rounded_rectangle((x0, y0, x0 + 590, y0 + 330), 12, fill=CARD, outline=RULE)
            d.text((x0 + 16, y0 + 12), LABEL[cell["cat"]], font=f(BOLD, 18), fill=INK)
            d.text((x0 + 574, y0 + 15), f"{len(cell['shared'])} shared codes", font=f(FONT, 13), fill=DIM, anchor="ra")
            for m, c in enumerate(cell["clips"]):
                cx, cy = x0 + 6 + m * 290 + 0, y0 + 42
                fr = c["frames"][min(i, len(c["frames"]) - 1)]
                im.paste(fr, (cx + 2, cy))
                seg = c["seg"]; t_now = min(i / FPS, seg["end"] - seg["start"])
                d.text((cx + 4, cy + 220), seg["task"].replace("_", " ")[:34], font=f(FONT, 12), fill=FAINT)
                n = len(seg["codes"]); bw = min(26, 284 / max(n, 1)); sx, sy = cx + 2, cy + 240
                for q, code in enumerate(seg["codes"]):
                    on = seg["times"][q] - seg["start"] <= t_now
                    base = cell["col"].get(code, GREY)
                    fill = base if on else tuple(int(v * 0.22 + 255 * 0.78) for v in base)
                    bx = sx + q * bw
                    d.rounded_rectangle((bx + 1, sy, bx + bw - 1, sy + 34), 4, fill=fill)
                    txt = (255, 255, 255) if on and code in cell["col"] else ((60, 70, 85) if on else (190, 198, 210))
                    d.text((bx + bw / 2, sy + 17), str(code), font=f(MONO, 10 if bw < 24 else 11), fill=txt, anchor="mm")
                px = sx + min(t_now / 0.5, n) * bw
                d.line((px, sy - 5, px, sy + 39), fill=INK, width=2)
        frames.append(im)
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as d:
        for i, im in enumerate(frames): im.save(f"{d}/{i:04d}.png")
        frames[min(len(frames) - 1, 3 * FPS)].save(OUT / "smt_motion_poster.png")
        src = ["-framerate", str(FPS), "-i", f"{d}/%04d.png"]
        subprocess.run(["ffmpeg", "-v", "error", "-y", *src, "-c:v", "libx264", "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(OUT / "smt_motion.mp4")], check=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", *src, "-c:v", "libvpx-vp9", "-crf", "34", "-b:v", "0", str(OUT / "smt_motion.webm")], check=False)
        subprocess.run(["ffmpeg", "-v", "error", "-y", *src, "-vf", "scale=1120:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=160[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
                        str(OUT / "smt_motion.gif")], check=True)
    print("frames", len(frames), {p.name: round(p.stat().st_size / 1e6, 2) for p in OUT.glob("smt_motion*")})


if __name__ == "__main__":
    main()
