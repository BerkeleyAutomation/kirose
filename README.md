# KiRoSe project page

Static project page for **KiRoSe: Using Kinematic Data for Efficient Segmentation of Robot Demonstrations**.
Everything is plain HTML/CSS/JS in `index.html`; there is no build step.

## Edit and preview
```bash
python3 -m http.server 8000   # then open http://localhost:8000
```
Open the page through a local server rather than as a file, because the video player loads `static/videos/segments.json`.

## Layout
| Path | What it is |
|---|---|
| `index.html` | The whole page: styles in `<style>`, the video player and title-figure scripts at the bottom |
| `static/images/` | Paper figures, the capabilities figure (`capabilities.png`), institution logos |
| `static/media/` | Animated Sparse Motion Token figure (`smt_motion.mp4/.webm/.gif`, poster) |
| `static/videos/` | Episode clips, `segments.json` (annotation / KiRoSe / video-VLM lanes) and the hero background `hero_mosaic.mp4` |
| `tools/` | Scripts that regenerated the media from the experiment outputs (need the private `traj_seg_vlm` data; not needed to edit the page) |

## Before release
- Paper / arXiv / Models buttons are disabled (`class="btn disabled"`); add the links and remove `disabled`.
- Replace the BibTeX with the arXiv entry.
- To host on GitHub Pages: make the repository public, then Settings → Pages → Deploy from branch `main`, folder `/`.
