"""Step 2 of the demo: add behaviours to the detections and render the demo video.

  python make_demo.py analyse            # behaviours + camera-compensated zones -> work/<scene>_beh.json
  python make_demo.py preview            # a few debug frames per scene (work/preview_*.jpg)
  python make_demo.py render             # final video -> farmpulse_demo.mp4

Detections/IDs come from the repo's YOLO11 + BoT-SORT + re-link pipeline (run_detection.py).
Behaviours come from the repo's rule engine (cow-detection/behaviour.py): standing at a feed bunk
for >= 3 s = eating, standing at a water trough for >= 2 s = drinking, otherwise standing.
"""
import bisect
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
CD = HERE.parent / "cow-detection"
sys.path.insert(0, str(CD))
sys.path.insert(0, str(HERE))

import behaviour as bh        # noqa: E402
from scenes import SCENES     # noqa: E402

WORK = HERE / "work"
OUT = HERE / "farmpulse_demo.mp4"
W, H, FPS = 1920, 1080, 30

# ---------------------------------------------------------------- analysis

COL = {"standing": (88, 196, 120), "eating": (255, 166, 43), "drinking": (64, 156, 255),
       "walking": (190, 190, 200), "lying": (170, 120, 255)}
LABEL = {"standing": "Standing", "eating": "Eating", "drinking": "Drinking"}


def camera_shift(video: Path, start_f: int, end_f: int):
    """Cumulative global image translation (source px) of every frame relative to start_f."""
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    out, prev, cum = {}, None, np.zeros(2)
    for idx in range(start_f, end_f):
        ok, f = cap.read()
        if not ok:
            break
        sc = 320 / f.shape[1]
        g = cv2.cvtColor(cv2.resize(f, None, fx=sc, fy=sc), cv2.COLOR_BGR2GRAY)
        if prev is not None:
            p = cv2.goodFeaturesToTrack(prev, 300, 0.01, 8)
            if p is not None:
                q, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, p, None)
                d = (q - p)[st.ravel() == 1].reshape(-1, 2)
                if len(d) > 10:
                    cum = cum + np.median(d, 0)
        out[idx] = cum / sc
        prev = g
    cap.release()
    return out


def shifted(poly, s):
    return np.array(poly, np.float32) + np.array(s, np.float32)


def analyse(sc):
    det = json.loads((WORK / f"{sc['name']}.json").read_text())
    fps, (w, h) = det["fps"], det["size"]
    video = CD / sc["video"]
    start_f = int(sc["start"] * fps)
    end_f = start_f + int(sc["dur"] * fps)
    shift = camera_shift(video, start_f, end_f)
    zones = sc["zones"]

    frames, aux = [], {}
    for fr in det["frames"]:
        s = shift.get(fr["f"], next(iter(shift.values())))
        pen = shifted(zones["pen"], s) if "pen" in zones else None
        masks = {}
        for t in ("feed", "water"):
            if t in zones:
                m = np.zeros((h, w), np.uint8)
                cv2.fillPoly(m, [shifted(zones[t], s).astype(np.int32)], 1)
                masks[t] = cv2.integral(m)
        keep, a = [], []
        for d in fr["d"]:
            if d[0] is None:
                continue
            x1, y1, x2, y2 = d[2:6]
            if (y2 - y1) < sc["min_h"] * h:
                continue
            if pen is not None and cv2.pointPolygonTest(pen, (float((x1 + x2) / 2), float(min(y2, h - 1))), False) < 0:
                continue
            keep.append(d)
            row = []
            for t in ("feed", "water"):
                ii = masks.get(t)
                if ii is None:
                    row.append(0.0)
                    continue
                X1, Y1, X2, Y2 = (int(max(0, min(w, x1))), int(max(0, min(h, y1))),
                                  int(max(0, min(w, x2))), int(max(0, min(h, y2))))
                area = (X2 - X1) * (Y2 - Y1)
                inside = ii[Y2, X2] - ii[Y1, X2] - ii[Y2, X1] + ii[Y1, X1] if area > 0 else 0
                row.append(float(inside) / area if area > 0 else 0.0)
            a.append(row + [None, None])
        frames.append({"f": fr["f"], "t": fr["t"], "d": keep})
        aux[fr["f"]] = a

    bh.ZONE_FRAC = sc.get("zone_frac", 0.12)
    bh.label_behaviours(frames, aux, fixed_camera=False)

    # tracks seen for at least 1.5 s only (drops one-frame flickers)
    seen = defaultdict(int)
    for fr in frames:
        for d in fr["d"]:
            seen[d[0]] += 1
    min_n = 1.5 * fps / det["stride"]
    ids = sorted(i for i, n in seen.items() if n >= min_n)
    for fr in frames:
        keep = [k for k, d in enumerate(fr["d"]) if d[0] in ids]
        fr["d"] = [fr["d"][k] for k in keep]
        fr["b"] = [fr["b"][k] for k in keep]
        fr["a"] = [aux[fr["f"]][k] for k in keep] if False else None

    # display smoothing: majority label over +-0.5 s per track
    per = defaultdict(list)
    for fi, fr in enumerate(frames):
        for k, d in enumerate(fr["d"]):
            per[d[0]].append((fr["t"], fi, k))
    for tid, obs in per.items():
        ts = np.array([o[0] for o in obs])
        labs = [frames[o[1]]["b"][o[2]] for o in obs]
        sm = []
        for i in range(len(obs)):
            lo, hi = np.searchsorted(ts, ts[i] - 0.5), np.searchsorted(ts, ts[i] + 0.5, side="right")
            win = labs[lo:hi]
            sm.append(max(set(win), key=win.count))
        for o, l in zip(obs, sm):
            frames[o[1]]["b"][o[2]] = l

    res = dict(name=sc["name"], fps=fps, size=[w, h], start_f=start_f, stride=det["stride"],
               ids=ids, frames=frames,
               shift={str(k): [float(v[0]), float(v[1])] for k, v in shift.items()})
    (WORK / f"{sc['name']}_beh.json").write_text(json.dumps(res))
    secs = defaultdict(lambda: defaultdict(float))
    dt = det["stride"] / fps
    for fr in frames:
        for d, l in zip(fr["d"], fr["b"]):
            secs[d[0]][l] += dt
    print(sc["name"], "cows:", len(ids))
    for i in ids:
        print("  cow", i, {k: round(v, 1) for k, v in secs[i].items()})
    return res


# ---------------------------------------------------------------- drawing helpers

def font(size, bold=False):
    return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size, index=1 if bold else 0)


F = {k: font(*v) for k, v in dict(h1=(64, True), h2=(34, True), h3=(26, True), body=(22, False),
                                  small=(18, False), tiny=(16, False), big=(54, True), label=(20, True),
                                  mono=(20, False)).items()}
BG, PANEL, LINE, TXT, MUTED, ACC = (12, 22, 18), (20, 34, 28), (44, 66, 56), (234, 242, 237), (140, 163, 151), (110, 226, 150)
VX, VY, VW, VH = 24, 104, 1248, 702  # video box on the canvas


def fmt_t(t):
    t = max(0.0, t)
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


def rounded(d, box, r, fill=None, outline=None, width=1):
    d.rounded_rectangle(box, r, fill=fill, outline=outline, width=width)


def dashed_poly(d, pts, color, width=3, dash=14):
    for i in range(len(pts)):
        (x1, y1), (x2, y2) = pts[i], pts[(i + 1) % len(pts)]
        L = max(1.0, float(np.hypot(x2 - x1, y2 - y1)))
        n = int(L // dash)
        for k in range(0, n, 2):
            a, b = k / n, min(1.0, (k + 1) / n)
            d.line([(x1 + (x2 - x1) * a, y1 + (y2 - y1) * a), (x1 + (x2 - x1) * b, y1 + (y2 - y1) * b)], fill=color, width=width)


def header(d, scene_title, t, idx, n_scenes):
    d.rectangle([0, 0, W, 84], fill=PANEL)
    d.line([0, 84, W, 84], fill=LINE, width=2)
    d.ellipse([28, 30, 52, 54], fill=ACC)
    d.text((66, 20), "FarmPulse AI", font=F["h2"], fill=TXT)
    d.text((300, 30), "Vision module  ·  cow detection and behaviour", font=F["body"], fill=MUTED)
    tag = f"Scene {idx}/{n_scenes}  ·  {scene_title}"
    tw = d.textlength(tag, font=F["h3"])
    rounded(d, (W - 68 - tw, 18, W - 28, 66), 24, fill=BG, outline=LINE, width=2)
    d.text((W - 48 - tw, 27), tag, font=F["h3"], fill=TXT)


def draw_scene_frame(canvas, sc, res, src, k, scene_state, n_scenes, sidx):
    """canvas: PIL RGB 1920x1080 (background already filled). src: BGR source frame."""
    fps, (w, h) = res["fps"], res["size"]
    t = k / FPS
    f_idx = res["start_f"] + int(round(t * fps))
    # video
    vid = cv2.resize(src, (VW, VH), interpolation=cv2.INTER_AREA)
    canvas.paste(Image.fromarray(cv2.cvtColor(vid, cv2.COLOR_BGR2RGB)), (VX, VY))
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    sx, sy = VW / w, VH / h
    s = res["shift"].get(str(min(max(f_idx, res["start_f"]), res["start_f"] + len(res["shift"]) - 1)), [0, 0])

    def tv(x, y):
        return (VX + x * sx, VY + y * sy)

    zcol = {"feed": (255, 166, 43), "water": (64, 156, 255)}
    zname = {"feed": "FEED BUNK ZONE", "water": "WATER TROUGH ZONE"}
    for zt in ("feed", "water"):
        if zt in sc["zones"]:
            pts = [tv(*p) for p in shifted(sc["zones"][zt], s)]
            d.polygon(pts, fill=zcol[zt] + (40,))
            dashed_poly(d, pts, zcol[zt] + (230,), 3)
            lx, ly = min(pts, key=lambda p: (p[1], p[0]))
            lx = min(max(lx, VX + 6), VX + VW - 230)
            ly = min(max(ly - 30, VY + 6), VY + VH - 40)
            rounded(d, (lx, ly, lx + d.textlength(zname[zt], font=F["tiny"]) + 20, ly + 28), 6, fill=zcol[zt] + (230,))
            d.text((lx + 10, ly + 5), zname[zt], font=F["tiny"], fill=(20, 20, 20))

    # detections of the latest processed record at/before this frame
    fr_f = res["_frame_ids"]
    j = bisect.bisect_right(fr_f, f_idx) - 1
    cur = res["frames"][max(j, 0)]
    counts = defaultdict(int)
    placed = []
    dets = sorted([(dd, lab) for dd, lab in zip(cur["d"], cur["b"]) if lab in COL], key=lambda z: z[0][2])
    for dd, lab in dets:
        counts[lab] += 1
        c = COL[lab]
        d.rounded_rectangle([tv(*dd[2:4]), tv(*dd[4:6])], 8, outline=c + (255,), width=4)
    crowded = len(dets) > 5
    for dd, lab in dets:
        tid, conf, x1, y1, x2, y2 = dd[:6]
        c = COL[lab]
        p1 = tv(x1, y1)
        txt = f"Cow {tid}  {LABEL.get(lab, lab)}" + ("" if crowded else f"  {conf * 100:.0f}%")
        tw = d.textlength(txt, font=F["label"])
        bx, by = p1[0], max(p1[1] - 34, VY + 4)
        bx = min(max(bx, VX + 4), VX + VW - tw - 22)
        for _ in range(8):  # nudge down until the label does not cover an earlier one
            if not any(bx < r[2] and bx + tw + 18 > r[0] and by < r[3] and by + 32 > r[1] for r in placed):
                break
            by += 34
        placed.append((bx, by, bx + tw + 18, by + 32))
        rounded(d, (bx, by, bx + tw + 18, by + 32), 8, fill=c + (245,))
        d.text((bx + 9, by + 5), txt, font=F["label"], fill=(14, 20, 16))
    # clip overlay to video box
    ov_c = ov.crop((VX, VY, VX + VW, VY + VH))
    canvas.paste(ov_c, (VX, VY), ov_c)
    d2 = ImageDraw.Draw(canvas, "RGBA")
    rounded(d2, (VX, VY, VX + VW, VY + VH), 2, outline=LINE, width=2)
    # timecode chip on video
    rounded(d2, (VX + 12, VY + VH - 46, VX + 232, VY + VH - 12), 8, fill=(0, 0, 0, 170))
    d2.ellipse([VX + 24, VY + VH - 36, VX + 38, VY + VH - 22], fill=(235, 70, 70))
    d2.text((VX + 48, VY + VH - 42), f"CAM-{sidx:02d}  {fmt_t(t)}", font=F["small"], fill=(255, 255, 255))
    header(d2, sc["title"], t, sidx, n_scenes)

    # ---------------- right panel
    px, py, pw = 1296, 104, 600
    rounded(d2, (px, py, px + pw, py + 702), 14, fill=PANEL, outline=LINE, width=2)
    d2.text((px + 24, py + 16), "LIVE ANALYSIS", font=F["small"], fill=MUTED)
    d2.text((px + 24, py + 42), str(len(cur["d"])), font=F["big"], fill=TXT)
    d2.text((px + 24 + d2.textlength(str(len(cur["d"])), font=F["big"]) + 14, py + 70), "cows detected now", font=F["body"], fill=MUTED)
    cw = (pw - 24 * 2 - 24) // 3
    for i, lab in enumerate(("standing", "eating", "drinking")):
        x0 = px + 24 + i * (cw + 12)
        rounded(d2, (x0, py + 124, x0 + cw, py + 204), 10, fill=BG, outline=COL[lab] if counts[lab] else LINE, width=2)
        d2.text((x0 + 14, py + 130), str(counts[lab]), font=F["h1"], fill=COL[lab] if counts[lab] else MUTED)
        d2.text((x0 + 14, py + 182 - 6 + 0), "", font=F["tiny"], fill=MUTED)
        d2.text((x0 + 14 + d2.textlength(str(counts[lab]), font=F["h1"]) + 10, py + 158), LABEL[lab], font=F["small"], fill=MUTED)

    # cow records
    d2.text((px + 24, py + 224), "COW RECORDS", font=F["small"], fill=MUTED)
    rows = sorted(zip(cur["d"], cur["b"]), key=lambda z: z[0][0])[:7]
    for i, (dd, lab) in enumerate(rows):
        y = py + 254 + i * 38
        d2.ellipse([px + 26, y + 8, px + 40, y + 22], fill=COL[lab])
        d2.text((px + 54, y + 2), f"Cow {dd[0]}", font=F["label"], fill=TXT)
        d2.text((px + 160, y + 2), LABEL.get(lab, lab), font=F["label"], fill=COL[lab])
        zone = "feed bunk" if lab == "eating" else "water trough" if lab == "drinking" else "pen"
        d2.text((px + 300, y + 4), f"zone: {zone}", font=F["small"], fill=MUTED)
        d2.text((px + pw - 24 - d2.textlength(f"{dd[1]:.2f}", font=F["small"]), y + 4), f"{dd[1]:.2f}", font=F["small"], fill=MUTED)
    if not rows:
        d2.text((px + 26, py + 258), "no cows in view", font=F["body"], fill=MUTED)

    # event log
    ey = py + 254 + 7 * 38 + 10
    d2.line([px + 24, ey - 6, px + pw - 24, ey - 6], fill=LINE, width=1)
    d2.text((px + 24, ey + 2), "EVENT LOG", font=F["small"], fill=MUTED)
    evs = [e for e in scene_state["events"] if e[0] <= t][-4:]
    for i, (et, tid, lab) in enumerate(evs):
        y = ey + 30 + i * 28
        d2.text((px + 24, y), fmt_t(et), font=F["mono"], fill=MUTED)
        d2.text((px + 110, y), f"Cow {tid}", font=F["mono"], fill=TXT)
        verb = {"eating": "eating at feed bunk", "drinking": "drinking at trough", "standing": "standing"}.get(lab, lab)
        d2.text((px + 200, y), verb, font=F["mono"], fill=COL[lab])

    # ---------------- bottom timeline
    ty = 830
    rounded(d2, (24, ty, W - 24, ty + 220), 14, fill=PANEL, outline=LINE, width=2)
    d2.text((48, ty + 14), "BEHAVIOUR TIMELINE PER COW", font=F["small"], fill=MUTED)
    lx = 1500
    for lab in ("standing", "eating", "drinking"):
        d2.rectangle([lx, ty + 18, lx + 18, ty + 36], fill=COL[lab])
        d2.text((lx + 26, ty + 14), LABEL[lab], font=F["small"], fill=TXT)
        lx += 130 + (10 if lab == "drinking" else 0)
    ids = scene_state["tl_ids"][:6]
    x0, x1 = 170, W - 60
    dur = sc["dur"]
    for r, tid in enumerate(ids):
        y = ty + 48 + r * 23
        d2.text((48, y - 1), f"Cow {tid}", font=F["small"], fill=TXT)
        d2.rectangle([x0, y + 1, x1, y + 17], fill=BG)
        for (a, b, lab) in scene_state["segments"][tid]:
            if a > t:
                break
            b = min(b, t)
            d2.rectangle([x0 + (x1 - x0) * a / dur, y + 1, x0 + (x1 - x0) * b / dur, y + 17], fill=COL[lab])
    xp = x0 + (x1 - x0) * t / dur
    d2.line([xp, ty + 44, xp, ty + 48 + max(len(ids), 1) * 23], fill=(255, 255, 255), width=2)
    foot = "YOLO11m detection  ·  BoT-SORT tracking + ID re-link  ·  rule-based behaviour from zone dwell time (not a trained behaviour model)"
    d2.text((48, ty + 200), foot, font=F["tiny"], fill=MUTED)


def scene_state(res, sc):
    """Event log + timeline segments from the labelled frames."""
    per = defaultdict(list)
    for fr in res["frames"]:
        for d, l in zip(fr["d"], fr["b"]):
            per[d[0]].append((fr["t"] - sc["start"], l))
    segs, events = {}, []
    gap = 1.0
    for tid, obs in per.items():
        s = []
        for t, l in obs:
            if s and s[-1][2] == l and t - s[-1][1] <= gap:
                s[-1][1] = t + res["stride"] / res["fps"]
            else:
                s.append([t, t + res["stride"] / res["fps"], l])
        segs[tid] = [tuple(x) for x in s]
        last = None
        for a, b, l in s:
            if l != last:
                events.append((a, tid, l))
                last = l
    events.sort()
    # only list behaviour changes worth showing (drop initial 'standing' spam when nothing else happens)
    events = [e for e in events if e[2] != "standing" or any(x[1] == e[1] and x[2] != "standing" and x[0] < e[0] for x in events)]
    ids = sorted(res["ids"], key=lambda i: -sum(b - a for a, b, _ in segs.get(i, [])))
    return dict(events=events, segments=segs, tl_ids=sorted(ids[:6]))


# ---------------------------------------------------------------- cards

def card_intro():
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im, "RGBA")
    d.ellipse([W // 2 - 360, 250, W // 2 - 320, 290], fill=ACC)
    d.text((W // 2 - 300, 230), "FarmPulse AI", font=font(110, True), fill=TXT)
    d.text((W // 2 - 360, 380), "Cow detection and behaviour monitoring", font=font(46), fill=TXT)
    d.text((W // 2 - 360, 450), "Vision module demo  ·  standing, eating, drinking", font=font(30), fill=MUTED)
    steps = [("Camera footage", ""), ("Detect cows", "YOLO11"), ("Track IDs", "BoT-SORT"), ("Behaviour", "zone rules"), ("Cow records", "dashboard")]
    x = 150
    for i, (a, b) in enumerate(steps):
        rounded(d, (x, 620, x + 280, 740), 16, fill=PANEL, outline=LINE, width=2)
        d.text((x + 24, 645), a, font=F["h3"], fill=TXT)
        d.text((x + 24, 690), b, font=F["body"], fill=ACC)
        if i < len(steps) - 1:
            d.polygon([(x + 292, 664), (x + 318, 680), (x + 292, 696)], fill=MUTED)
        x += 340
    d.text((150, 940), "Mockup built from the project's own footage and code. Behaviour labels are rule-based, not a trained model.", font=F["small"], fill=MUTED)
    return im


def card_outro(summary):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im, "RGBA")
    d.text((120, 90), "What the system recorded", font=font(64, True), fill=TXT)
    d.text((120, 180), "Cow-seconds per behaviour in each demo scene (all cows combined)", font=font(30), fill=MUTED)
    y = 290
    for title, secs in summary:
        tot = sum(secs.values()) or 1
        rounded(d, (120, y, W - 120, y + 170), 16, fill=PANEL, outline=LINE, width=2)
        d.text((150, y + 18), title, font=F["h2"], fill=TXT)
        d.text((150, y + 66), f"{tot:.0f} cow-seconds observed", font=F["body"], fill=MUTED)
        x0, x1 = 700, W - 160
        xx = x0
        for lab in ("standing", "eating", "drinking"):
            v = secs.get(lab, 0)
            if v > 0:
                wv = (x1 - x0) * v / tot
                d.rectangle([xx, y + 40, xx + wv - 3, y + 100], fill=COL[lab])
                if wv > 90:
                    d.text((xx + 12, y + 56), f"{LABEL[lab]} {100 * v / tot:.0f}%", font=F["label"], fill=(14, 20, 16))
                xx += wv
        y += 200
    d.text((120, 930), "Next: fine-tune the detector on farm footage  ·  train a behaviour classifier  ·  cow re-identification  ·  suspicion model and alerts", font=F["body"], fill=MUTED)
    return im


# ---------------------------------------------------------------- render

class Writer:
    def __init__(self, path):
        import imageio_ffmpeg
        self.p = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                   "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                                   "-crf", "18", "-preset", "medium", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)

    def write(self, im):
        self.p.stdin.write(np.asarray(im.convert("RGB")).tobytes())

    def close(self):
        self.p.stdin.close()
        self.p.wait()


def fade(im, a):
    return Image.blend(Image.new("RGB", (W, H), BG), im, max(0.0, min(1.0, a)))


def load_res(sc):
    res = json.loads((WORK / f"{sc['name']}_beh.json").read_text())
    res["_frame_ids"] = [f["f"] for f in res["frames"]]
    return res


def render(only=None, out=OUT):
    wr = Writer(out)
    summary = []
    FADE = int(0.35 * FPS)
    intro = card_intro()
    for k in range(int(3.2 * FPS)):
        wr.write(fade(intro, min(k / FADE, (int(3.2 * FPS) - k) / FADE)))
    for si, sc in enumerate(SCENES, 1):
        if only and sc["name"] not in only:
            continue
        res = load_res(sc)
        st = scene_state(res, sc)
        secs = defaultdict(float)
        for fr in res["frames"]:
            for l in fr["b"]:
                secs[l] += res["stride"] / res["fps"]
        summary.append((sc["title"], dict(secs)))
        cap = cv2.VideoCapture(str(CD / sc["video"]))
        fps = res["fps"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, res["start_f"])
        cache, nxt, last = {}, res["start_f"], None
        n = int(sc["dur"] * FPS)
        for k in range(n):
            target = res["start_f"] + int(round(k / FPS * fps))
            while nxt <= target:
                ok, fr_ = cap.read()
                if ok:
                    last = fr_
                nxt += 1
            canvas = Image.new("RGB", (W, H), BG)
            draw_scene_frame(canvas, sc, res, last, k, st, len(SCENES), si)
            wr.write(fade(canvas, min(k / FADE, (n - k) / FADE)))
        cap.release()
        print("rendered", sc["name"], flush=True)
    outro = card_outro(summary)
    for k in range(int(5 * FPS)):
        wr.write(fade(outro, min(k / FADE, (int(5 * FPS) - k) / FADE)))
    wr.close()
    print("wrote", out)


def preview():
    for sc in SCENES:
        res = load_res(sc)
        st = scene_state(res, sc)
        cap = cv2.VideoCapture(str(CD / sc["video"]))
        for frac in (0.1, 0.45, 0.85):
            k = int(frac * sc["dur"] * FPS)
            cap.set(cv2.CAP_PROP_POS_FRAMES, res["start_f"] + int(round(k / FPS * res["fps"])))
            ok, src = cap.read()
            canvas = Image.new("RGB", (W, H), BG)
            draw_scene_frame(canvas, sc, res, src, k, st, len(SCENES), SCENES.index(sc) + 1)
            canvas.save(WORK / f"preview_{sc['name']}_{int(frac * 100)}.jpg", quality=88)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "render"
    if cmd == "analyse":
        for sc in SCENES:
            if (WORK / f"{sc['name']}.json").exists():
                analyse(sc)
    elif cmd == "preview":
        preview()
    else:
        render()
