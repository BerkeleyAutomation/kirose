"""Title-section video (1920x1080): what the released KiRoSe models do.
  1 title: a filmstrip of the episode (one frame per segment) over its KiRoSe timeline  2 the sole input is kinematic state (camera crossed out; zoomed, labelled kinematic channels)
  3 annotation of one AgiBot G1 episode: reference video, the live kinematic input and KiRoSe's timed annotations
  4 accuracy versus cost per demonstration  5 released models.
All kinematics and annotations shown are real: episode 655509 and its KiRoSe output from static/videos/segments.json.
The robot camera clips are the dataset's 640x480, 5 fps preview videos, so they are softer than the graphics.

  ~/traj_seg_vlm/.venv/bin/python tools/hero_video.py              -> static/media/hero.{mp4,webm}, hero_poster.jpg
  ~/traj_seg_vlm/.venv/bin/python tools/hero_video.py kinonly:4 ... -> preview stills in the scratchpad
"""
import json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

SITE = Path(__file__).resolve().parent.parent; ROOT = Path.home() / "traj_seg_vlm"; OUT = SITE / "static/media"
SCRATCH = Path("/tmp/claude-1000/-home-shadeform/8d80cac8-7129-4e38-9d6e-386829a91191/scratchpad")
FPS, W, H, S, OUTW, OUTH = 15, 1280, 720, 3, 1920, 1080      # layout in 1280x720 units, drawn at 3x, saved at 1080p
FD = SITE / "tools/fonts"                                    # Newsreader (headings), Inter (text), IBM Plex Mono (labels)
FONT, BOLD, SERIF, MONO, MONOM = "inter400", "inter600", "serif", FD / "IBMPlexMono-Regular.ttf", FD / "IBMPlexMono-Medium.ttf"
_FC = {}
def f(p, s):
    k = (p, s)
    if k not in _FC:
        if p == SERIF: fo = ImageFont.truetype(str(FD / "Newsreader.ttf"), int(s * S)); fo.set_variation_by_axes([400, 72])
        elif p in (FONT, BOLD): fo = ImageFont.truetype(str(FD / "Inter.ttf"), int(s * S)); fo.set_variation_by_axes([min(max(s, 14), 32), 400 if p == FONT else 600])
        else: fo = ImageFont.truetype(str(p), int(s * S))
        _FC[k] = fo
    return _FC[k]
BG, INK, DIM, FAINT, RULE = (238, 238, 236), (24, 24, 27), (82, 82, 88), (132, 132, 138), (224, 224, 222)
ACC, ORG, RED, TEAL, HL, PANEL = (52, 98, 238), (235, 104, 52), (214, 48, 49), (14, 150, 166), (234, 239, 253), (255, 255, 255)
FLORA, OPUS, ASTRA = (194, 24, 122), (27, 175, 122), (235, 104, 52)
WHITE = (255, 255, 255)
SCENES = [("title", 6.0), ("kinonly", 7.0), ("pipeline", 15.0), ("cost", 8.0), ("models", 6.5)]
FADE = 0.4
EP = "655509"
# six readable channels: linear speed, angular speed and gripper opening rate for each arm
CH = [("Left arm · linear speed", 0, "lin", INK), ("Left arm · angular speed", 0, "ang", INK), ("Left gripper · opening rate", 0, "grip", INK),
      ("Right arm · linear speed", 7, "lin", ORG), ("Right arm · angular speed", 7, "ang", ORG), ("Right gripper · opening rate", 7, "grip", ORG)]


def ease(x): x = min(max(x, 0), 1); return 1 - (1 - x) ** 3
def T(d, xy, s, font, fill, anchor="la"): d.text((xy[0] * S, xy[1] * S), s, font=font, fill=fill, anchor=anchor)
def R(d, x0, y0, x1, y1, r=8, **k): d.rounded_rectangle((x0 * S, y0 * S, x1 * S, y1 * S), radius=r * S, **k)
def line(d, pts, **k): d.line([(x * S, y * S) for x, y in pts], **k)
def mix(a, b, t): t = min(max(t, 0), 1); return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))
def mmss(x): m = int(x // 60); return f"{m}:{x - 60 * m:04.1f}"


def fit(img, w, h):                                   # cover-crop to w x h layout units
    W0, H0 = img.size; sc = max(w * S / W0, h * S / H0)
    im = img.resize((int(W0 * sc + 1), int(H0 * sc + 1)), Image.LANCZOS)
    x, y = (im.width - w * S) // 2, (im.height - h * S) // 2
    return im.crop((x, y, x + w * S, y + h * S))


def paste_round(im, src, x, y, r=12):
    m = Image.new("L", src.size, 0); ImageDraw.Draw(m).rounded_rectangle((0, 0, src.width - 1, src.height - 1), radius=r * S, fill=255)
    im.paste(src, (int(x * S), int(y * S)), m)


LOGOS = {}
def paste_logo(im, name, x, y, size, alpha=1.0):
    if name not in LOGOS: LOGOS[name] = Image.open(SITE / f"tools/logos/{name}.png").convert("RGBA")
    lg = LOGOS[name].resize((int(size * S), int(size * S)), Image.LANCZOS)
    if alpha < 1: lg.putalpha(lg.getchannel("A").point(lambda v: int(v * alpha)))
    im.paste(lg, (int(x * S), int(y * S)), lg)


def pill(d, x, y, s, bg, fg=WHITE, size=13, anchor="l"):
    fo = f(MONOM, size); w = d.textlength(s, font=fo) / S + 28
    if anchor == "m": x -= w / 2
    R(d, x, y, x + w, y + size + 16, r=(size + 16) / 2, fill=bg); T(d, (x + 14, y + 7), s, fo, fg)


def frames_of(path):
    with tempfile.TemporaryDirectory() as d:          # 5 fps source, blended up to the video frame rate
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"minterpolate=fps={FPS}:mi_mode=blend", f"{d}/%05d.png"], check=True)
        return [Image.open(p).convert("RGB") for p in sorted(Path(d).glob("*.png"))]


def load():
    seg = json.load(open(SITE / "static/videos/segments.json")); main = next(r for g in seg["groups"] for r in g["runs"] if r["id"] == EP)
    z = np.load(ROOT / f"data/cache_signed2_agibot/{EP}.npz"); X = np.concatenate([z["feat_left"], z["feat_right"]], axis=1)
    sm = lambda v: np.convolve(v, np.ones(7) / 7, "same")
    chans = []
    for _, o, kind, _ in CH:
        v = np.linalg.norm(X[:, o:o + 3], axis=1) if kind == "lin" else np.linalg.norm(X[:, o + 3:o + 6], axis=1) if kind == "ang" else X[:, o + 6]
        v = sm(v); chans.append(v / (np.percentile(np.abs(v), 99.5) + 1e-6))
    return dict(main=main, t=z["t"], ch=np.array(chans), frames=frames_of(SITE / main["video"]))


def scope(d, D, x, y, w, h, a, b, labels=True, lw=2.4, playhead=None, alpha=1.0):
    """Six labelled kinematic channels over the time window [a, b], one row each, thick enough to read."""
    t, C = D["t"], D["ch"]; lab = 214 if labels else 0; pw = w - lab; rh = h / len(CH); sel = (t >= a) & (t <= b)
    for k, (name, _, kind, col) in enumerate(CH):
        y0 = y + k * rh; mid = y0 + rh / 2 if kind == "grip" else y0 + rh * 0.9
        if labels: T(d, (x, y0 + rh / 2 - 8), name, f(FONT, 13), mix(BG, DIM, alpha))
        line(d, [(x + lab, mid), (x + w, mid)], fill=mix(BG, RULE, alpha), width=S)
        if sel.sum() > 1:
            amp = rh * (0.42 if kind == "grip" else 0.8)
            pts = [(x + lab + pw * (tt - a) / (b - a), mid - amp * np.clip(v, -1, 1)) for tt, v in zip(t[sel], C[k, sel])]
            line(d, pts, fill=mix(BG, col, alpha), width=int(lw * S), joint="curve")
    if playhead is not None:
        px = x + lab + pw * (playhead - a) / (b - a); line(d, [(px, y - 4), (px, y + h + 4)], fill=TEAL, width=2 * S)


# ---------- scenes ----------
VERB = {"pick": (52, 98, 238), "place": (235, 104, 52), "move": (14, 150, 166), "adjust": (139, 92, 214), "reset": (150, 150, 155)}


def s_title(im, d, t, dur, D):
    """A filmstrip of the episode above its KiRoSe timeline: one frame per segment, the timeline filling in behind a playhead."""
    segs, Tep, fr = D["main"]["kirose"], D["main"]["duration"], D["frames"]
    a = ease(t / 0.7)
    T(d, (W / 2, 84), "KiRoSe", f(SERIF, 84), mix(BG, INK, a), "ma")
    T(d, (W / 2, 196), "segments robot demonstrations into timed subtasks", f(MONO, 19), mix(BG, DIM, ease((t - 0.3) / 0.7)), "ma")
    n, gap, x0, x1 = len(segs), 12, 64, W - 64
    fw = (x1 - x0 - (n - 1) * gap) / n; fh = fw * 0.75; fy, ty, th = 284, 486, 40
    X = lambda tt: x0 + (x1 - x0) * tt / Tep
    sweep = Tep * min(max((t - 1.4) / 3.2, 0), 1)       # constant-speed playhead
    for k, g in enumerate(segs):
        verb = g["d"].split()[0].lower(); col = VERB.get(verb, FAINT); fx = x0 + k * (fw + gap)
        p = ease((t - 0.6 - 0.12 * k) / 0.5)
        if p > 0:                                       # frame from the middle of the segment
            img = fit(fr[min(int((g["s"] + g["e"]) / 2 * FPS), len(fr) - 1)], fw, fh)
            img = Image.blend(Image.new("RGB", img.size, BG), img, p); paste_round(im, img, fx, fy, r=8)
        q = ease((sweep - g["s"]) / 1.2) if sweep > g["s"] else 0
        if q > 0:
            sx0, sx1 = X(g["s"]) + 1.5, X(min(g["e"], max(sweep, g["s"]))) - 1.5
            if sx1 > sx0: R(d, sx0, ty, sx1, ty + th, r=6, fill=mix(BG, col, 0.85))
            R(d, fx - 3, fy - 3, fx + fw + 3, fy + fh + 3, r=10, outline=mix(BG, col, q), width=3 * S)
            cx = (X(g["s"]) + X(g["e"])) / 2
            line(d, [(fx + fw / 2, fy + fh + 4), (fx + fw / 2, fy + fh + 26), (cx, ty - 24), (cx, ty - 4)], fill=mix(BG, col, 0.6 * q), width=2 * S, joint="curve")
            T(d, (cx, ty + th + 12), verb, f(MONOM, 14), mix(BG, col, q), "ma")
    R(d, x0, ty, x1, ty + th, r=6, outline=RULE, width=S)
    if 0 < sweep < Tep:
        px = X(sweep); line(d, [(px, ty - 10), (px, ty + th + 6)], fill=INK, width=2 * S)
    for tt in range(0, int(Tep) + 1, 10):
        T(d, (X(tt), ty + th + 44), f"{tt} s", f(MONO, 13), mix(BG, FAINT, a), "ma")
    c = ease((t - 4.6) / 0.6)
    if c > 0: T(d, (W / 2, 636), "AgiBot G1 · Toast bread · kinematic input only", f(MONO, 15), mix(BG, FAINT, c), "ma")


def s_kinonly(im, d, t, dur, D):
    a = ease(t / 0.6); Tep = D["main"]["duration"]
    T(d, (W / 2, 36), "The sole model input is the robot's kinematic state", f(SERIF, 34), mix(BG, INK, a), "ma")
    T(d, (W / 2, 88), "No camera images are used. KiRoSe reads end-effector velocities and gripper motion recorded by the robot.",
      f(FONT, 17), mix(BG, DIM, a), "ma")
    # left: camera, crossed out
    vx, vy, vw, vh = 48, 150, 400, 300
    img = fit(D["frames"][int(16 * FPS)], vw, vh); g = ease((t - 0.8) / 0.6)
    if g > 0: img = Image.blend(Image.blend(img, ImageOps.grayscale(img).convert("RGB"), g), Image.new("RGB", img.size, BG), 0.55 * g)
    paste_round(im, img, vx, vy, r=16)
    if g > 0:
        cx, cy, k = vx + vw / 2, vy + vh / 2, 62 * g
        line(d, [(cx - k, cy - k), (cx + k, cy + k)], fill=RED, width=10 * S); line(d, [(cx - k, cy + k), (cx + k, cy - k)], fill=RED, width=10 * S)
    T(d, (vx + vw / 2, vy + vh + 16), "Camera video", f(BOLD, 19), DIM, "ma")
    T(d, (vx + vw / 2, vy + vh + 42), "not used", f(BOLD, 19), mix(BG, RED, g), "ma")
    # right: full-episode overview, a magnified window, and the zoomed channels
    kx, kw = 500, W - 48 - 500; b = ease((t - 0.4) / 0.6)
    T(d, (kx, 150), f"Kinematic state, full episode ({Tep:.0f} s)", f(BOLD, 14), mix(BG, DIM, b))
    scope(d, D, kx, 172, kw, 96, 0, Tep * ease((t - 0.4) / 1.6), labels=False, lw=1.0, alpha=b)
    za, zb = 18.5, 26.5; z = ease((t - 2.0) / 0.8)     # 8 s window in which both arms move
    if z > 0:
        x0, x1 = kx + kw * za / Tep, kx + kw * zb / Tep
        R(d, x0, 166, x1, 274, r=4, outline=mix(BG, ACC, z), width=2 * S)
        line(d, [(x0, 274), (kx, 296)], fill=mix(BG, ACC, z), width=S); line(d, [(x1, 274), (kx + kw, 296)], fill=mix(BG, ACC, z), width=S)
        R(d, kx, 296, kx + kw, 616, r=14, fill=mix(BG, PANEL, z), outline=mix(BG, ACC, z), width=3 * S)
        pill(d, kx + 14, 308, f"ZOOM · {za:.0f}–{zb:.0f} s", mix(BG, ACC, z), size=12)
        scope(d, D, kx + 18, 346, kw - 36, 258, za, za + (zb - za) * ease((t - 2.4) / 2.2), lw=3.0, alpha=z)
    c = ease((t - 4.6) / 0.5)
    if c > 0: pill(d, W / 2, 640, "Input: kinematic state only", mix(BG, ACC, c), size=24, anchor="m")


def s_pipeline(im, d, t, dur, D):
    r = D["main"]; Tep = r["duration"]
    play = min(max((t - 0.8) / (dur - 2.0), 0), 1) * Tep
    T(d, (W / 2, 30), "Annotating a robot demonstration from kinematic state", f(SERIF, 30), INK, "ma")
    T(d, (W / 2, 76), f'AgiBot G1, task "{r["instruction"]}", a robot not seen in training', f(FONT, 16), DIM, "ma")
    vx, vy, vw, vh = 48, 116, 560, 315
    fr = D["frames"]; img = fit(fr[min(int(play * FPS), len(fr) - 1)], vw, vh)
    paste_round(im, img, vx, vy, r=14)
    pill(d, vx + 12, vy + 12, f"{mmss(play)} / {mmss(Tep)}", (250, 250, 250), INK, size=12)
    pill(d, vx + 12, vy + vh - 40, "Video shown for reference only; not a model input", (250, 250, 250), INK, size=12)
    # the model input: a scrolling, magnified window of the kinematic channels
    iy = vy + vh + 14; ih = 720 - 28 - iy
    R(d, vx, iy, vx + vw, iy + ih, r=14, fill=PANEL, outline=ACC, width=3 * S)
    pill(d, vx + 12, iy + 10, "MODEL INPUT: KINEMATIC STATE", ACC, size=12)
    T(d, (vx + vw - 14, iy + 15), "last 8 s", f(FONT, 12), FAINT, "ra")
    scope(d, D, vx + 14, iy + 44, vw - 28, ih - 54, play - 8, play, lw=2.6, playhead=play)
    # the model output
    ox, oy, ow = 640, 116, W - 48 - 640
    pill(d, ox, oy, "MODEL OUTPUT: TIMED SUBTASK ANNOTATIONS", INK, size=12)
    for k, g in enumerate(r["kirose"]):
        y = oy + 44 + k * 62
        if play < g["s"]: break
        on = g["s"] <= play < g["e"]; a = ease((play - g["s"]) / 1.0)
        if on: R(d, ox, y - 4, ox + ow, y + 50, r=10, fill=HL)
        desc = g["d"].rstrip("."); desc = desc[0].upper() + desc[1:]
        T(d, (ox + 16, y + 2), f'{mmss(g["s"])} – {mmss(g["e"])}', f(MONO, 13), mix(BG, ACC if on else FAINT, a))
        T(d, (ox + 16, y + 21), desc, f(BOLD if on else FONT, 18), mix(BG, INK if on else DIM, a))


def s_cost(im, d, t, dur, D):
    a = ease(t / 0.6)
    T(d, (W / 2, 30), "Accuracy versus cost per demonstration", f(SERIF, 32), mix(BG, INK, a), "ma")
    T(d, (W / 2, 80), "Boundary F1 within ±1 s on 50 held-out YAM and 20 AgiBot G1 episodes; cost per demonstration on a logarithmic scale",
      f(FONT, 16), mix(BG, DIM, a), "ma")
    L, Rr, Tp, B = 190, 1150, 150, 590; lo, hi, ylo, yhi = np.log10(6e-5), np.log10(3), 0.45, 0.82
    X = lambda c: L + (Rr - L) * (np.log10(c) - lo) / (hi - lo); Y = lambda v: B - (B - Tp) * (v - ylo) / (yhi - ylo)
    for e in range(-4, 1):
        for m in range(1, 10):
            c = m * 10.0 ** e
            if 6e-5 < c < 3: line(d, [(X(c), Tp), (X(c), B)], fill=mix(BG, (236, 241, 248) if m > 1 else RULE, a), width=S)
        T(d, (X(10.0 ** e), B + 12), ["$0.0001", "$0.001", "$0.01", "$0.10", "$1"][e + 4], f(MONO, 14), mix(BG, FAINT, a), "ma")
    for v in (0.5, 0.6, 0.7, 0.8):
        line(d, [(L, Y(v)), (Rr, Y(v))], fill=mix(BG, RULE, a), width=S); T(d, (L - 12, Y(v) - 9), f"{v:.1f}", f(MONO, 14), mix(BG, FAINT, a), "ra")
    line(d, [(L, B), (Rr, B)], fill=mix(BG, FAINT, a), width=S)
    T(d, ((L + Rr) / 2, B + 40), "Cost per demonstration (USD, log scale)", f(FONT, 16), mix(BG, DIM, a), "ma")
    T(d, (L - 70, (Tp + B) / 2), "Boundary F1 ±1 s", f(FONT, 16), mix(BG, DIM, a), "ma")
    # most expensive first, then right to left down to KiRoSe; each model is labelled with its maker's logo
    M = [("GPT-6 Astra", ASTRA, 0.60, 0.54, 1.01, "openai", "ra"), ("Claude Opus 5.5", OPUS, 0.50, 0.67, 0.33, "claude", "ra"),
         ("Flora", FLORA, 0.583, 0.774, 0.0077, "nvidia", "la"), ("KiRoSe (ours)", ACC, 0.666, 0.772, 0.000156, "berkeley_seal", "la")]
    for k, (n, col, yam, agi, c, logo, anc) in enumerate(M):
        p = ease((t - 0.6 - 0.55 * k) / 0.5)
        if p <= 0: continue
        x, r = X(c), 11 * p
        d.ellipse(((x - r) * S, (Y(yam) - r) * S, (x + r) * S, (Y(yam) + r) * S), fill=col)
        d.ellipse(((x - r) * S, (Y(agi) - r) * S, (x + r) * S, (Y(agi) + r) * S), outline=col, width=4 * S, fill=BG)
        top = max(yam, agi); ly = Y(top) - 44 if k == 3 else Y(top) - 15; fo = f(BOLD, 18)
        tw = d.textlength(n, font=fo) / S; lg = 30
        lx = x + 18 if anc == "la" else x - 18 - tw - lg - 8      # [logo] name, beside the upper point
        paste_logo(im, logo, lx, ly + 15 - lg / 2, lg, p)
        T(d, (lx + lg + 8, ly + 3), n, fo, mix(BG, col, p))
    q = ease((t - 3.2) / 0.6)
    if q > 0:                                          # cost brackets
        y = Tp - 8; x0, x1 = X(0.000156), X(0.0077)
        line(d, [(x0, y + 8), (x0, y), (x1, y), (x1, y + 8)], fill=mix(BG, FLORA, q), width=2 * S)
        T(d, ((x0 + x1) / 2, y - 26), "≈50× lower cost than Flora", f(BOLD, 18), mix(BG, INK, q), "ma")
        y = B - 26; x2, x3 = X(0.000156), X(0.33)
        line(d, [(x2, y - 8), (x2, y), (x3, y), (x3, y - 8)], fill=mix(BG, DIM, q), width=2 * S)
        T(d, ((x2 + x3) / 2, y - 30), "≈2,000–6,500× lower cost than frontier API models", f(BOLD, 18), mix(BG, INK, q), "ma")
    s = ease((t - 0.4) / 0.6); ly = 676
    d.ellipse(((L) * S, (ly) * S, (L + 14) * S, (ly + 14) * S), fill=mix(BG, DIM, s))
    T(d, (L + 22, ly - 2), "YAM, held-out tasks", f(FONT, 15), mix(BG, DIM, s))
    d.ellipse(((L + 230) * S, ly * S, (L + 244) * S, (ly + 14) * S), outline=mix(BG, DIM, s), width=3 * S)
    T(d, (L + 252, ly - 2), "AgiBot G1 (zero-shot for KiRoSe)", f(FONT, 15), mix(BG, DIM, s))
    T(d, (Rr, ly - 2), "KiRoSe: one H200 GPU at $3.50 per hour", f(FONT, 15), mix(BG, FAINT, s), "ra")


def s_models(im, d, t, dur, D):
    a = ease(t / 0.8)
    T(d, (W / 2, 110), "Released models", f(SERIF, 38), mix(BG, INK, a), "ma")
    T(d, (W / 2, 168), "Model weights and loading code are available on Hugging Face.", f(FONT, 20), mix(BG, DIM, a), "ma")
    cards = [("KiRoSe-SMT", "Sparse Motion Token encoder", "2.8M parameters"), ("KiRoSe-Reader", "Annotation model", "9B parameters"),
             ("KiRoSe-Progress", "Progress estimation head", "68k parameters")]
    cw, gap = 340, 30; x0 = (W - 3 * cw - 2 * gap) / 2
    for k, (n, s, p) in enumerate(cards):
        al = ease((t - 0.4 - 0.25 * k) / 0.6); x = x0 + k * (cw + gap)
        R(d, x, 230, x + cw, 360, r=14, fill=BG, outline=mix(BG, RULE, al), width=2 * S)
        T(d, (x + cw / 2, 254), n, f(BOLD, 25), mix(BG, ACC, al), "ma"); T(d, (x + cw / 2, 296), s, f(FONT, 18), mix(BG, INK, al), "ma")
        T(d, (x + cw / 2, 326), p, f(MONO, 14), mix(BG, FAINT, al), "ma")
    c = ease((t - 2.2) / 0.8)
    T(d, (W / 2, 470), "KiRoSe: Towards a Kinematic Foundation Model", f(SERIF, 26), mix(BG, INK, c), "ma")
    T(d, (W / 2, 516), "berkeleyautomation.github.io/kirose", f(MONO, 19), mix(BG, DIM, c), "ma")


def render(name, t, dur, D):
    im = Image.new("RGB", (W * S, H * S), BG); d = ImageDraw.Draw(im)
    globals()["s_" + name](im, d, t, dur, D)
    im = im.resize((OUTW, OUTH), Image.LANCZOS)
    a = min(1, t / FADE, (dur - t) / FADE)                     # fade through white between scenes
    return Image.blend(Image.new("RGB", (OUTW, OUTH), BG), im, max(a, 0)) if a < 1 else im


def main():
    D = load(); OUT.mkdir(parents=True, exist_ok=True); n = 0
    with tempfile.TemporaryDirectory() as tmp:
        for name, dur in SCENES:
            for i in range(int(dur * FPS)):
                im = render(name, i / FPS, dur, D); im.save(f"{tmp}/{n:05d}.png"); n += 1
                if name == "title" and i == int(5.5 * FPS): im.save(OUT / "hero_poster.jpg", quality=90)
        src = ["-framerate", str(FPS), "-i", f"{tmp}/%05d.png"]
        subprocess.run(["ffmpeg", "-v", "error", "-y", *src, "-c:v", "libx264", "-crf", "18", "-preset", "slow", "-tune", "animation",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(OUT / "hero.mp4")], check=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", *src, "-c:v", "libvpx-vp9", "-crf", "28", "-b:v", "0", "-row-mt", "1", str(OUT / "hero.webm")], check=True)
    print("ok", n, "frames")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        D = load()
        for a in sys.argv[1:]:
            nm, tt = a.split(":"); render(nm, float(tt), dict(SCENES)[nm], D).save(SCRATCH / f"hv_{nm}_{tt}.png")
    else:
        main()
