"""Title-section video: one explainer of what the released KiRoSe model does.
  1 title  2 robot motion in -> motion tokens -> timed events out (AgiBot "Toast bread", real kinematics, codes and output)
  3 the same model on YAM, AgiBot G1 and human hands  4 training data (tools/train_data_video.py)  5 accuracy and cost
  6 download and run.  All outputs shown are the model's real outputs from static/videos/segments.json.

  ~/traj_seg_vlm/.venv/bin/python tools/hero_video.py   -> static/media/hero.{mp4,webm}, hero_poster.jpg
"""
import json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import matplotlib; matplotlib.use("Agg"); from matplotlib import cm

SITE = Path(__file__).resolve().parent.parent; ROOT = Path.home() / "traj_seg_vlm"; OUT = SITE / "static/media"
sys.path.insert(0, str(SITE / "tools")); import train_data_video as TD
FPS, W, H, S = 15, 1280, 720, 2
FONT, BOLD = TD.FONT, TD.BOLD; MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
f = lambda p, s: ImageFont.truetype(p, int(s * S))
BG, INK, DIM, FAINT, CARD, RULE, ACC, ORG = TD.BG, TD.INK, TD.DIM, TD.FAINT, TD.CARD, TD.RULE, TD.ACC, TD.ORG
BG = (255, 255, 255)                                  # white video background
RED = (220, 38, 38)
TEAL, HL, ELEV = (14, 165, 181), (226, 236, 252), (236, 243, 253)
FLORA, OPUS, ASTRA = (194, 24, 122), (27, 175, 122), (235, 104, 52)
LIGHT = (147, 197, 253)
ease = TD.ease
SCENES = [("title", 4.0), ("kinonly", 5.5), ("pipeline", 15.0), ("download", 6.5)]
FADE = 0.4


# ---------- drawing helpers (coordinates in output pixels) ----------
def T(d, xy, s, font, fill, anchor="la"): d.text((xy[0] * S, xy[1] * S), s, font=font, fill=fill, anchor=anchor)
def R(d, x0, y0, x1, y1, r=8, **k): d.rounded_rectangle((x0 * S, y0 * S, x1 * S, y1 * S), radius=r * S, **k)
def box(d, x0, y0, x1, y1, **k): d.rectangle((x0 * S, y0 * S, x1 * S, y1 * S), **k)
def line(d, pts, **k): d.line([(x * S, y * S) for x, y in pts], **k)
def mix(a, b, t): return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))
def mmss(x): m = int(x // 60); return f"{m}:{x - 60 * m:04.1f}"


def wrap(d, s, font, width):
    out, cur = [], ""
    for w in s.split():
        nxt = (cur + " " + w).strip()
        if d.textlength(nxt, font=font) <= width * S or not cur: cur = nxt
        else: out.append(cur); cur = w
    return out + [cur] if cur else out


def fit(img, w, h):                                   # cover-crop to w x h output pixels
    W0, H0 = img.size; sc = max(w * S / W0, h * S / H0)
    im = img.resize((int(W0 * sc + 1), int(H0 * sc + 1)), Image.BILINEAR)
    x, y = (im.width - w * S) // 2, (im.height - h * S) // 2
    return im.crop((x, y, x + w * S, y + h * S))


def paste_round(im, src, x, y, r=12):
    m = Image.new("L", src.size, 0); ImageDraw.Draw(m).rounded_rectangle((0, 0, src.width - 1, src.height - 1), radius=r * S, fill=255)
    im.paste(src, (int(x * S), int(y * S)), m)


def frames_of(path, fps=5):
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"fps={fps}", f"{d}/%05d.png"], check=True)
        return [Image.open(p).convert("RGB") for p in sorted(Path(d).glob("*.png"))]


def chip(d, x, y, s, fg=(255, 255, 255), bg=(11, 18, 32), size=12):
    fo = f(MONO, size); w = d.textlength(s, font=fo) / S
    R(d, x, y, x + w + 18, y + size + 12, r=(size + 12) / 2, fill=bg); T(d, (x + 9, y + 5), s, fo, fg)


def header(d, step, title, sub=None, a=1.0):
    c = mix(BG, ACC, a); T(d, (48, 34), step, f(MONO, 13), c)
    T(d, (48, 54), title, f(BOLD, 30), mix(BG, INK, a))
    if sub: T(d, (48, 96), sub, f(FONT, 16), mix(BG, DIM, a))


# ---------- data ----------
def load():
    seg = json.load(open(SITE / "static/videos/segments.json")); runs = {r["id"]: r for g in seg["groups"] for r in g["runs"]}
    main = runs["655509"]
    z = np.load(ROOT / "data/cache_signed2_agibot/655509.npz")
    kin = [(z["t"], np.convolve(np.linalg.norm(z[f"feat_{a}"][:, :3], axis=1), np.ones(5) / 5, "same")) for a in ("left", "right")]
    kin14 = (z["t"], np.concatenate([z["feat_left"], z["feat_right"]], axis=1))
    codes = json.load(open(ROOT / "kin_codes_signed_trainonly/655509.json"))
    bodies = [("YAM", "held-out task", runs["episode_abe6954f-1a1e-46e6-a410-a2fce60a1374"], 8.0),
              ("AgiBot G1", "zero-shot robot", runs["648680"], 2.0),
              ("Human hands", "zero-shot, Ego-Exo4D", runs["37081591-1f5f-454f-b2a4-bfdcec5ccca6_r55"], 4.0)]
    vids = {r["id"]: frames_of(SITE / r["video"]) for r in [main] + [b[2] for b in bodies]}
    return dict(main=main, kin=kin, kin14=kin14, codes=codes, bodies=bodies, vids=vids, td=TD.load())


# ---------- scenes: each takes (im, d, t, dur, D) ----------
def s_title(im, d, t, dur, D):
    a, b, c = ease(t / 0.8), ease((t - 0.6) / 0.8), ease((t - 1.4) / 0.8)
    T(d, (W / 2, 250), "KiRoSe", f(BOLD, 96), mix(BG, ACC, a), "ma")
    T(d, (W / 2, 376), "Towards Kinematic Foundation Model", f(BOLD, 40), mix(BG, INK, b), "ma")
    T(d, (W / 2, 450), "A paper and an open model", f(FONT, 24), mix(BG, DIM, c), "ma")


def traces(d, D, x, y, w, h, upto, rows=14):
    """The 14 kinematic channels as stacked sparklines (left arm ink, right arm orange), drawn up to time `upto`."""
    tt, X = D["kin14"]; Tep = D["main"]["duration"]; sel = tt <= upto
    if sel.sum() < 2: return
    rh = h / rows
    for c in range(rows):
        v = X[sel, c][::2]; ts = tt[sel][::2]; m = np.abs(X[:, c]).max() + 1e-6; yc = y + (c + 0.5) * rh
        line(d, [(x + w * a / Tep, yc - 0.45 * rh * b / m) for a, b in zip(ts, v)], fill=INK if c < 7 else ORG, width=int(1.3 * S))


def s_kinonly(im, d, t, dur, D):
    a = ease(t / 0.6)
    T(d, (W / 2, 48), "KiRoSe never sees the video", f(BOLD, 40), mix(BG, INK, a), "ma")
    T(d, (W / 2, 106), "Its only input is the robot's kinematic state: end-effector velocities and gripper motion, 14 channels.",
      f(FONT, 19), mix(BG, DIM, a), "ma")
    # left: camera video, crossed out
    vx, vy, vw, vh = 96, 170, 480, 360
    fr = D["vids"][D["main"]["id"]][60]; img = fit(fr, vw, vh)
    g = ease((t - 0.9) / 0.6)
    if g > 0: img = Image.blend(img, ImageOps.grayscale(img).convert("RGB"), g); img = Image.blend(img, Image.new("RGB", img.size, BG), 0.55 * g)
    paste_round(im, img, vx, vy, r=16)
    if g > 0:
        cx, cy, k = vx + vw / 2, vy + vh / 2, 70 * g
        line(d, [(cx - k, cy - k), (cx + k, cy + k)], fill=RED, width=10 * S); line(d, [(cx - k, cy + k), (cx + k, cy - k)], fill=RED, width=10 * S)
    T(d, (vx + vw / 2, vy + vh + 18), "Camera video", f(BOLD, 20), DIM, "ma")
    T(d, (vx + vw / 2, vy + vh + 46), "not used", f(BOLD, 20), mix(BG, RED, g), "ma")
    # right: kinematic traces, the only input
    kx, ky, kw, kh = 704, 170, 480, 360; b = ease((t - 0.4) / 0.6)
    R(d, kx, ky, kx + kw, ky + kh, r=16, fill=mix(BG, (239, 245, 255), b), outline=mix(BG, ACC, b), width=3 * S)
    traces(d, D, kx + 20, ky + 16, kw - 40, kh - 32, D["main"]["duration"] * ease((t - 0.5) / 2.5))
    T(d, (kx + kw / 2, ky + kh + 18), "Kinematic state", f(BOLD, 20), mix(BG, INK, b), "ma")
    T(d, (kx + kw / 2, ky + kh + 46), "the only input", f(BOLD, 20), mix(BG, ACC, b), "ma")
    c = ease((t - 2.6) / 0.5)
    if c > 0:
        R(d, W / 2 - 330, 626, W / 2 + 330, 690, r=32, fill=mix(BG, ACC, c))
        T(d, (W / 2, 640), "Only kinematic state is used", f(BOLD, 28), mix(ACC, (255, 255, 255), c), "ma")


def s_pipeline(im, d, t, dur, D):
    r = D["main"]; Tep = r["duration"]
    play = min(max((t - 0.8) / (dur - 2.0), 0), 1) * Tep
    T(d, (W / 2, 44), "Annotate robot data with kinematic state only", f(BOLD, 30), INK, "ma")
    vx, vy, vw, vh = 48, 120, 600, 360
    fr = D["vids"][r["id"]]; img = fit(fr[min(int(play * 5), len(fr) - 1)], vw, vh)
    paste_round(im, Image.blend(img, Image.new("RGB", img.size, BG), 0.15), vx, vy, r=16)
    chip(d, vx + 14, vy + 14, f'{r["instruction"].upper()}  ·  {mmss(play)}', size=12)
    chip(d, vx + 14, vy + vh - 40, "VIDEO: FOR YOU TO WATCH, NOT USED BY THE MODEL", size=11)
    # the actual model input, highlighted
    ix, iy, ih = vx, vy + vh + 16, 170
    R(d, ix, iy, ix + vw, iy + ih, r=14, fill=(239, 245, 255), outline=ACC, width=3 * S)
    R(d, ix + 14, iy + 12, ix + 318, iy + 40, r=14, fill=ACC)
    T(d, (ix + 28, iy + 18), "MODEL INPUT: KINEMATIC STATE ONLY", f(BOLD, 13), (255, 255, 255))
    traces(d, D, ix + 16, iy + 48, vw - 32, ih - 60, play)
    ox, oy, ow = 680, 120, W - 48 - 680
    R(d, ox, oy, ox + 318, oy + 28, r=14, fill=INK)
    T(d, (ox + 14, oy + 6), "OUTPUT: TIMED ANNOTATIONS", f(BOLD, 13), (255, 255, 255))
    for k, g in enumerate(r["kirose"]):
        y = oy + 44 + k * 56
        if play < g["s"]: break
        on = g["s"] <= play < g["e"]; a = ease((play - g["s"]) / 1.0)
        if on: R(d, ox, y, ox + ow, y + 46, r=10, fill=HL)
        T(d, (ox + 16, y + 6), f'{mmss(g["s"])} – {mmss(g["e"])}', f(MONO, 12), mix(BG, ACC if on else FAINT, a))
        T(d, (ox + 16, y + 22), g["d"].rstrip("."), f(BOLD if on else FONT, 16), mix(BG, INK if on else DIM, a))


def s_bodies(im, d, t, dur, D):
    header(d, "ONE MODEL", "The same model on other robots and on human hands", "Trained on one robot (YAM). AgiBot G1 and human hands are zero-shot.")
    tw, gap = (W - 96 - 2 * 28) / 3, 28
    for k, (name, note, r, t0) in enumerate(D["bodies"]):
        a = ease((t - 0.2 * k) / 0.6); x = 48 + k * (tw + gap); y = 140
        play = t0 + max(t - 0.2 * k, 0) * 3.0
        fr = D["vids"][r["id"]]; paste_round(im, fit(fr[min(int(play * 5), len(fr) - 1)], int(tw), 300), x, y)
        if a < 1: im.paste(Image.blend(im.crop((int(x * S), int(y * S), int((x + tw) * S), int((y + 300) * S))), Image.new("RGB", (int(tw * S), 300 * S), BG), 1 - a), (int(x * S), int(y * S)))
        chip(d, x + 12, y + 12, name.upper(), size=11)
        T(d, (x, y + 314), note, f(FONT, 13), FAINT)
        seg = next((g for g in r["kirose"] if g["s"] <= play < g["e"]), None)
        R(d, x, y + 340, x + tw, y + 440, r=10, fill=CARD, outline=RULE, width=S)
        T(d, (x + 16, y + 352), "KIROSE", f(MONO, 11), ACC)
        if seg:
            T(d, (x + tw - 16, y + 352), f'{mmss(seg["s"])}–{mmss(seg["e"])}', f(MONO, 11), FAINT, "ra")
            for j, ln in enumerate(wrap(d, seg["d"].rstrip("."), f(BOLD, 16), tw - 32)[:2]): T(d, (x + 16, y + 374 + j * 22), ln, f(BOLD, 16), INK)
        Tep = r["duration"]; ly = y + 456
        for j, g in enumerate(r["kirose"]):
            box(d, x + tw * g["s"] / Tep + 0.5, ly, x + tw * g["e"] / Tep - 0.5, ly + 12, fill=ACC if j % 2 == 0 else LIGHT)
        px = x + tw * min(play, Tep) / Tep; line(d, [(px, ly - 4), (px, ly + 16)], fill=TEAL, width=2 * S)


def s_data(im, d, t, dur, D):
    fr = TD.frame(min(int(t * 10), TD.TOTAL - 1), *D["td"]); im.paste(fr.resize((W * S, H * S), Image.LANCZOS), (0, 0))


def s_cost(im, d, t, dur, D):
    header(d, "ACCURACY AND COST", "As accurate as video models, at a fraction of the cost",
           "Boundary F1 within ±1 s on 50 held-out YAM and 20 AgiBot G1 episodes; cost per demonstration")
    M = [("KiRoSe", ACC, 0.666, 0.772, 0.000156, "$0.00016"), ("Flora", FLORA, 0.583, 0.774, 0.0077, "$0.0077"),
         ("GPT-6 Astra", ASTRA, 0.60, 0.54, 1.01, "$1.01"), ("Claude Opus 5.5", OPUS, 0.50, 0.67, 0.33, "$0.33")]
    p = ease(t / 1.6)
    # accuracy
    x0, y0, bw = 48, 160, 560
    T(d, (x0, y0), "Boundary F1 ±1 s  (higher is better)", f(BOLD, 17), INK)
    for gi, (gname, col) in enumerate((("YAM, held-out tasks", 2), ("AgiBot G1, zero-shot for KiRoSe", 3))):
        gy = y0 + 42 + gi * 210; T(d, (x0, gy), gname, f(FONT, 14), DIM)
        for k, m in enumerate(M):
            y = gy + 26 + k * 40; v = m[col]; L = (bw - 160) * v / 0.8 * p
            T(d, (x0, y + 6), m[0], f(BOLD if k == 0 else FONT, 14), INK)
            R(d, x0 + 130, y, x0 + 130 + max(L, 2), y + 28, r=5, fill=m[1])
            T(d, (x0 + 138 + L, y + 6), f"{v * min(p * 1.3, 1):.2f}", f(BOLD if k == 0 else FONT, 14), INK)
    # cost, log scale
    cx, cw = 690, W - 48 - 690 - 110
    T(d, (cx, y0), "Cost per demonstration  (log scale)", f(BOLD, 17), INK)
    lo, hi = -4.2, 0.3; Xc = lambda c: cx + 130 + cw * (np.log10(c) - lo) / (hi - lo)
    for e in range(-4, 1):
        x = Xc(10 ** e); line(d, [(x, y0 + 60), (x, y0 + 252)], fill=RULE, width=S)
        T(d, (x, y0 + 258), ["$0.0001", "$0.001", "$0.01", "$0.10", "$1"][e + 4], f(MONO, 11), FAINT, "ma")
    for k, m in enumerate(M):
        y = y0 + 68 + k * 46; xe = Xc(10 ** (lo + (np.log10(m[4]) - lo) * p))
        T(d, (cx, y + 6), m[0], f(BOLD if k == 0 else FONT, 14), INK)
        R(d, cx + 130, y, max(xe, cx + 132), y + 28, r=5, fill=m[1])
        T(d, (xe + 8, y + 6), m[5], f(BOLD if k == 0 else FONT, 14), INK)
    a = ease((t - 1.8) / 0.8)
    if a > 0:
        R(d, cx, y0 + 300, W - 48, y0 + 430, r=12, fill=mix(BG, CARD, a), outline=mix(BG, RULE, a), width=S)
        T(d, (cx + 22, y0 + 320), "≈50× cheaper than Flora", f(BOLD, 26), mix(BG, ACC, a))
        T(d, (cx + 22, y0 + 360), "≈2,000–6,500× cheaper than frontier API models", f(BOLD, 18), mix(BG, INK, a))
        T(d, (cx + 22, y0 + 392), "KiRoSe: 0.16 s per demonstration on one H200 GPU", f(FONT, 14), mix(BG, DIM, a))


CODE = [("from transformers import AutoModel", 0), ("from vllm import LLM, SamplingParams", 0), ("", 0),
        ('smt = AutoModel.from_pretrained("sy-xie/KiRoSe-SMT", trust_remote_code=True)', 0),
        ("enc = smt.encode(kin, t)            # (T, 14) end-effector rates", 0),
        ('reader = LLM("sy-xie/KiRoSe-Reader")', 0),
        ('out = reader.generate([smt.reader_prompt(enc, "Toast bread")], params)', 0),
        ("events = smt.parse_segments(out[0].outputs[0].text, enc.duration)", 0),
        ('# [{"start": 0.1, "end": 7.4, "description": "Pick up the bread ..."}, ...]', 1)]


def s_download(im, d, t, dur, D):
    a = ease(t / 0.8)
    T(d, (W / 2, 120), "Open models, ready to download", f(BOLD, 34), mix(BG, INK, a), "ma")
    cards = [("KiRoSe-SMT", "motion tokenizer"), ("KiRoSe-Reader", "writes timed events"), ("KiRoSe-Progress", "task progress")]
    cw, gap = 330, 30; x0 = (W - 3 * cw - 2 * gap) / 2
    for k, (n, s) in enumerate(cards):
        al = ease((t - 0.4 - 0.25 * k) / 0.6); x = x0 + k * (cw + gap)
        R(d, x, 210, x + cw, 320, r=14, fill=mix(BG, CARD, al), outline=mix(BG, RULE, al), width=S)
        T(d, (x + cw / 2, 238), n, f(BOLD, 24), mix(BG, ACC, al), "ma"); T(d, (x + cw / 2, 278), s, f(FONT, 17), mix(BG, DIM, al), "ma")
    b = ease((t - 1.6) / 0.8)
    T(d, (W / 2, 400), "Hugging Face  ·  sy-xie/KiRoSe", f(MONO, 20), mix(BG, INK, b), "ma")
    c = ease((t - 2.4) / 0.8)
    T(d, (W / 2, 520), "KiRoSe — Towards Kinematic Foundation Model", f(BOLD, 26), mix(BG, INK, c), "ma")
    T(d, (W / 2, 566), "berkeleyautomation.github.io/kirose", f(MONO, 18), mix(BG, DIM, c), "ma")


def render(name, t, dur, D):
    im = Image.new("RGB", (W * S, H * S), BG); d = ImageDraw.Draw(im)
    globals()["s_" + name](im, d, t, dur, D)
    im = im.resize((W, H), Image.LANCZOS)
    a = min(1, t / FADE, (dur - t) / FADE)                     # fade through the page background
    return Image.blend(Image.new("RGB", (W, H), BG), im, max(a, 0)) if a < 1 else im


def main():
    D = load(); OUT.mkdir(parents=True, exist_ok=True); n = 0
    with tempfile.TemporaryDirectory() as tmp:
        for name, dur in SCENES:
            for i in range(int(dur * FPS)):
                im = render(name, i / FPS, dur, D); im.save(f"{tmp}/{n:05d}.png"); n += 1
                if name == "pipeline" and i == int(9 * FPS): im.convert("RGB").save(OUT / "hero_poster.jpg", quality=88)
        src = ["-framerate", str(FPS), "-i", f"{tmp}/%05d.png"]
        subprocess.run(["ffmpeg", "-v", "error", "-y", *src, "-c:v", "libx264", "-crf", "23", "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(OUT / "hero.mp4")], check=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", *src, "-c:v", "libvpx-vp9", "-crf", "36", "-b:v", "0", "-row-mt", "1", str(OUT / "hero.webm")], check=True)
    print("ok", n, "frames")


if __name__ == "__main__":
    if len(sys.argv) > 1:                                     # preview stills: hero_video.py scene:t ...
        D = load()
        for a in sys.argv[1:]:
            nm, tt = a.split(":"); render(nm, float(tt), dict(SCENES)[nm], D).save(f"/tmp/claude-1000/-home-shadeform/8d80cac8-7129-4e38-9d6e-386829a91191/scratchpad/hv_{nm}_{tt}.png")
    else:
        main()
