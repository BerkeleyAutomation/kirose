"""Progress panel of the title figure: one AgiBot G1 subtask (zero-shot) played as recorded and then reversed in place,
with the KiRoSe progress head's curve for each. Uses capabilities_fig.progress (frozen SMT encoder + signed-rate head). CPU.

  CUDA_VISIBLE_DEVICES="" ~/traj_seg_vlm/.venv/bin/python tools/progress_demo.py --ep 655509 --unit 1
  -> static/media/progress_demo.mp4 (forward then reversed), static/media/progress_demo.json
"""
import argparse, json, subprocess, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from capabilities_fig import progress, ROOT, SITE


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ep", default="655509"); ap.add_argument("--split", default="test"); ap.add_argument("--unit", type=int, default=1)
    a = ap.parse_args()
    rec = next(r for r in map(json.loads, open(ROOT / f"runs_signed_reader/20260923_trainonly/preds_agibot_{a.split}.jsonl")) if r["episode_id"] == a.ep)
    u = sorted(rec["gt_segments"], key=lambda s: s["start"])[a.unit]
    tt, prog, _, _ = progress(a.ep, f"agibot_{a.split}", u["start"], u["end"])
    out = SITE / "static/media"; D = u["end"] - u["start"]
    json.dump(dict(episode=a.ep, instruction=rec["instruction"], subtask=u.get("description", ""), duration=round(D, 3),
                   t=[round(float(x), 3) for x in tt], forward=[round(float(x), 4) for x in prog["forward"]],
                   reversed=[round(float(x), 4) for x in prog["reversed"]]), open(out / "progress_demo.json", "w"))
    with tempfile.TemporaryDirectory() as d:
        src = ["-ss", f"{u['start']:.3f}", "-t", f"{D:.3f}", "-i", str(ROOT / f"data/agibot_video/{a.ep}.mp4")]
        vf = "fps=10,scale=480:-2"
        subprocess.run(["ffmpeg", "-v", "error", "-y", *src, "-vf", vf, "-an", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", f"{d}/f.mp4"], check=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{d}/f.mp4", "-vf", "reverse", "-an", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", f"{d}/r.mp4"], check=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{d}/f.mp4", "-i", f"{d}/r.mp4", "-filter_complex", "[0][1]concat=n=2:v=1[v]", "-map", "[v]",
                        "-c:v", "libx264", "-crf", "26", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out / "progress_demo.mp4")], check=True)
    print("ok", u, D, prog["forward"][-1], prog["reversed"][-1])


if __name__ == "__main__":
    main()
