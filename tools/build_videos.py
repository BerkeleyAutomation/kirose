"""Build the video results for the KiRoSe page: compact web MP4s plus one JSON of synced segment lanes.

Episodes: the paper's appendix picks (appendix_kf/picks_*.json) first, then more chosen with the paper's
own rule (plot_seg_examples.select: round-robin over tasks, episode closest to the task's median KiRoSe
BF1 at 1 s; not score-maximising). Human hands reuse the 8 Ego-Exo4D clips of the Hand Segmentation Viewer.

  python3 tools/build_videos.py --per 8 --handviewer <dir with viewer_data.json and clips/>
  -> static/videos/{agibot,yam,human}/<id>.mp4, static/videos/segments.json
"""
import argparse, json, statistics, subprocess
from collections import defaultdict
from pathlib import Path

SITE = Path(__file__).resolve().parents[1]
TSV = Path.home() / "traj_seg_vlm"
RUNS = TSV / "runs_signed_reader"
OUT = SITE / "static/videos"


def load(*paths):
    d = {}
    for p in paths:
        d.update({r["episode_id"]: r for r in map(json.loads, open(p))})
    return d


def bf1(pred, gt, tol=1.0):      # same greedy one-to-one matcher as the paper's figure script
    b = lambda segs: sorted(s["start"] for s in sorted(segs, key=lambda s: s["start"])[1:])
    pb, gb = b(pred), b(gt)
    if not pb and not gb:
        return 1.0
    used, tp = set(), 0
    for p in pb:
        best, bd = None, tol
        for i, g in enumerate(gb):
            if i not in used and abs(p - g) <= bd:
                best, bd = i, abs(p - g)
        if best is not None:
            used.add(best); tp += 1
    pr, rc = (tp / len(pb) if pb else 0), (tp / len(gb) if gb else 0)
    return 2 * pr * rc / (pr + rc) if pr + rc else 0.0


def select(ours, base, n):
    by_task = defaultdict(list)
    for e in ours:
        if e in base:
            by_task[ours[e]["task"]].append(e)
    queues = {}
    for t, eps in by_task.items():
        sc = {e: bf1(ours[e]["pred_segments"], ours[e]["gt_segments"]) for e in eps}
        med = statistics.median(sc.values())
        queues[t] = sorted(eps, key=lambda e: (abs(sc[e] - med), e))
    tasks = sorted(queues, key=lambda t: (-len(queues[t]), t))
    picks = []
    while len(picks) < n and any(queues[t] for t in tasks):
        for t in tasks:
            if queues[t] and len(picks) < n:
                picks.append(queues[t].pop(0))
    return picks


def lane(segs):
    return [dict(s=round(s["start"], 2), e=round(s["end"], 2), d=s.get("description", "")) for s in sorted(segs, key=lambda s: s["start"])]


def encode(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", "scale='min(640,iw)':-2", "-c:v", "libx264",
                        "-preset", "slow", "-crf", "27", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(dst)], check=True)


def robot(name, ours, base, picks_file, video_of, per, instr_of):
    order = [p["episode_id"] for p in json.load(open(picks_file)) if p["episode_id"] in ours and p["episode_id"] in base]
    for e in select(ours, base, per * 3):
        if len(order) >= per:
            break
        if e not in order:
            order.append(e)
    runs = []
    for e in order[:per]:
        o, b = ours[e], base[e]
        encode(video_of(e), OUT / name / f"{e}.mp4")
        runs.append(dict(id=e, embodiment=name, task=o["task"], instruction=instr_of(o), duration=round(o["duration_s"], 2),
                         video=f"static/videos/{name}/{e}.mp4", paper_figure=e in {p["episode_id"] for p in json.load(open(picks_file))},
                         gt=lane(o["gt_segments"]), kirose=lane(o["pred_segments"]), vlm=lane(b["pred_segments"]),
                         f1=dict(kirose=round(bf1(o["pred_segments"], o["gt_segments"]), 3), vlm=round(bf1(b["pred_segments"], o["gt_segments"]), 3))))
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=8)
    ap.add_argument("--handviewer", type=Path, required=True)
    a = ap.parse_args()
    kf = TSV / "appendix_kf"
    pretty = lambda t: t.replace("_", " ").capitalize()

    ag_ours = load(RUNS / "20260923_trainonly/preds_agibot_test.jsonl", RUNS / "20260923_trainonly/preds_agibot_ood.jsonl")
    ag_base = load(RUNS / "baseline_video_zeroshot/preds_agibot_test_evalset.jsonl", RUNS / "baseline_video_zeroshot/preds_agibot_ood.jsonl")
    agibot = robot("agibot", ag_ours, ag_base, kf / "picks_agibot.json", lambda e: TSV / f"data/agibot_video/{e}.mp4", a.per,
                   lambda o: o.get("instruction", ""))

    yam_ours = load(RUNS / "20260923_trainonly/preds_yam_eval.jsonl")
    yam_base = load(RUNS / "baseline_video_zeroshot/preds_yam_eval.jsonl")
    yam = robot("yam", yam_ours, yam_base, kf / "picks_yam.json", lambda e: TSV / f"proc/{yam_ours[e]['task']}/{e}/video.mp4", a.per,
                lambda o: o.get("instruction") or pretty(o["task"]))

    human = []
    for r in json.load(open(a.handviewer / "viewer_data.json"))["runs"]:
        encode(a.handviewer / "clips" / f"{r['id']}.mp4", OUT / "human" / f"{r['id']}.mp4")
        human.append(dict(id=r["id"], embodiment="human", task=r["task"], instruction=r["task"], duration=r["duration"],
                          video=f"static/videos/human/{r['id']}.mp4", paper_figure=False,
                          gt=r["gt"], kirose=r["smt"], vlm=r["vlm"], f1=dict(kirose=r["f1"]["smt"], vlm=r["f1"]["vlm"])))

    data = dict(note="KiRoSe = SMT reader trained on YAM only (runs_signed_reader/20260923_trainonly, long2 smt_correct for hands); "
                     "vlm = Qwen3.5-9B zero-shot on video + instruction; f1 = boundary F1 at +-1 s vs the annotation.",
                groups=[dict(key="agibot", label="AgiBot G1", note="zero-shot embodiment", runs=agibot),
                        dict(key="yam", label="YAM", note="held-out tasks", runs=yam),
                        dict(key="human", label="Human hands", note="zero-shot, Ego-Exo4D", runs=human)])
    (OUT / "segments.json").write_text(json.dumps(data, separators=(",", ":")))
    for g in data["groups"]:
        print(g["key"], len(g["runs"]), [(r["task"][:24], r["f1"]["kirose"], r["f1"]["vlm"], r["paper_figure"]) for r in g["runs"]])


if __name__ == "__main__":
    main()
