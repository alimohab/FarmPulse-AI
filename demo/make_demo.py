"""Step 2 of the demo: add behaviours to the detections and render the explained demo video.

  python make_demo.py analyse            # behaviours + camera-compensated zones -> work/<scene>_beh.json
  python make_demo.py preview            # debug frames (normal + pause frames) -> work/preview_*.jpg
  python make_demo.py render             # final video -> farmpulse_demo_long.mp4

Detections/IDs come from the repo's YOLO11 + BoT-SORT + re-link pipeline (run_detection.py).
Behaviours come from the repo's rule engine (cow-detection/behaviour.py): standing at a feed bunk
for >= 3 s = eating, standing at a water trough for >= 2 s = drinking, otherwise standing.
The video also shows the dwell timer (CHECKING) that runs before a cow counts as eating/drinking.
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
OUT = HERE / "farmpulse_demo_long.mp4"
W, H, FPS = 1920, 1080, 30
NEED = {"eating": bh.MIN_EAT_S, "drinking": bh.MIN_DRINK_S}   # seconds at the trough before the label counts
INTRO_S, EVENT_S, HOLD_S = 5.0, 4.0, 4.0

GREEN, ORANGE, BLUE, YELLOW = (88, 196, 120), (255, 166, 43), (64, 156, 255), (250, 215, 70)
STATE_COL = {"standing": GREEN, "eating": ORANGE, "drinking": BLUE, "pend_eating": YELLOW, "pend_drinking": YELLOW}
STATE_NAME = {"standing": "Standing", "eating": "Eating", "drinking": "Drinking",
              "pend_eating": "Checking", "pend_drinking": "Checking"}
ZONE_TXT = {"standing": "in pen", "eating": "at feed bunk", "drinking": "at water trough",
            "pend_eating": "at feed bunk", "pend_drinking": "at water trough"}

# ---------------------------------------------------------------- analysis


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


def display_states(frames, dt):
    """Causal display state per detection: a cow at a trough shows CHECKING (timer) until the dwell
    time is reached, then EATING / DRINKING. Sets fr['s'] = [[state, seconds_in_state_run], ...]."""
    per = defaultdict(list)
    for fi, fr in enumerate(frames):
        fr["s"] = [["standing", 0.0] for _ in fr["d"]]
        for k, d in enumerate(fr["d"]):
            per[d[0]].append((fr["t"], fi, k, fr["b"][k]))
    for tid, obs in per.items():
        run_start, prev_lab, prev_t = None, None, None
        for t, fi, k, lab in obs:
            if lab in NEED:
                if prev_lab != lab or t - prev_t > 1.0:
                    run_start = t
                dwell = t - run_start
                st = lab if dwell >= NEED[lab] - 1e-6 else "pend_" + lab
                frames[fi]["s"][k] = [st, round(dwell, 2)]
            else:
                run_start = None
            prev_lab, prev_t = lab, t


def analyse(sc):
    det = json.loads((WORK / f"{sc['name']}.json").read_text())
    fps, (w, h) = det["fps"], det["size"]
    video = CD / sc["video"]
    start_f = int(sc["start"] * fps)
    end_f = start_f + int(sc["dur"] * fps)
    ref_f = int(sc.get("ref", sc["start"]) * fps)
    shift = camera_shift(video, start_f, end_f)
    base = shift.get(ref_f, next(iter(shift.values())))
    shift = {k: v - base for k, v in shift.items()}
    zones = sc["zones"]

    frames, aux = [], {}
    for fr in det["frames"]:
        s = shift.get(fr["f"], np.zeros(2))
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

    # display smoothing: majority label over +-0.8 s per track
    per = defaultdict(list)
    for fi, fr in enumerate(frames):
        for k, d in enumerate(fr["d"]):
            per[d[0]].append((fr["t"], fi, k))
    for tid, obs in per.items():
        ts = np.array([o[0] for o in obs])
        labs = [frames[o[1]]["b"][o[2]] for o in obs]
        sm = []
        for i in range(len(obs)):
            lo, hi = np.searchsorted(ts, ts[i] - 0.8), np.searchsorted(ts, ts[i] + 0.8, side="right")
            win = labs[lo:hi]
            sm.append(max(set(win), key=win.count))
        for o, l in zip(obs, sm):
            frames[o[1]]["b"][o[2]] = l

    display_states(frames, det["stride"] / fps)
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


# ---------------------------------------------------------------- plan: segments, events, pauses, captions

def cows_txt(ids):
    ids = sorted(ids)
    names = [str(i) for i in ids]
    if len(names) == 1:
        return f"Cow {names[0]}"
    return "Cows " + ", ".join(names[:-1]) + " and " + names[-1]


def scene_plan(res, sc):
    per = defaultdict(list)
    for fr in res["frames"]:
        for d, s, l in zip(fr["d"], fr["s"], fr["b"]):
            per[d[0]].append((fr["t"] - sc["start"], s[0], d[1], l))
    dt = res["stride"] / res["fps"]
    segs, events = {}, []
    for tid, obs in per.items():
        s, last = [], None
        for t, st, conf, _ in obs:
            if s and s[-1][2] == st and t - s[-1][1] <= 1.0:
                s[-1][1] = t + dt
            else:
                s.append([t, t + dt, st])
        segs[tid] = [tuple(x) for x in s]
        for a, b, st in s:
            if st == last:
                continue
            if st == "standing" and last is None:
                last = st
                continue
            typ = {"pend_eating": "reach_feed", "eating": "eat", "pend_drinking": "reach_water",
                   "drinking": "drink", "standing": "leave"}[st]
            events.append((a, tid, typ))
            last = st
    events.sort()
    ids = sorted(res["ids"], key=lambda i: -sum(b - a for a, b, _ in segs.get(i, [])))
    plan = dict(events=events, segments=segs, tl_ids=sorted(ids[:6]), per=per)

    # explainer pauses
    flags, pauses = sc.get("pauses", []), []
    if "detect" in flags:
        pauses.append(dict(t=1.2, ids="all", text="Detection and tracking. Every box is one cow found by YOLO11, with its confidence in %. "
                           "The ID (Cow 1, Cow 2 ...) comes from the tracker and stays with the cow from frame to frame."))
    for kind, typ, secs, zone in (("eat", "eat", "3", "feed bunk"), ("drink", "drink", "2", "water trough")):
        if kind in flags:
            ev = [e for e in events if e[2] == typ and e[0] > 0.5]
            if ev:
                t0, tid, _ = ev[0]
                name = "EATING" if kind == "eat" else "DRINKING"
                pauses.append(dict(t=t0 + 1.5 * dt, ids=[tid], text=f"Cow {tid} has now stood at the {zone} for {secs} seconds, "
                                   f"so its label changes from CHECKING to {name}. A cow that leaves before {secs} s stays STANDING."))
    if "stand" in flags:
        best = None
        for tid, obs in per.items():
            if all(st == "standing" for _, st, _, _ in obs) and len(obs) * dt >= 4.0:
                score = float(np.mean([c for _, _, c, _ in obs]))
                if best is None or score > best[0]:
                    best = (score, tid, obs)
        if best:
            _, tid, obs = best
            tm = obs[len(obs) // 2][0]
            pauses.append(dict(t=tm, ids=[tid], text=f"Cow {tid} is inside the pen but not at a trough, so it is simply STANDING. "
                               "For now the system uses three states: standing, eating and drinking."))
    pauses.sort(key=lambda p: p["t"])
    out = []
    for p in pauses:
        if not out or p["t"] - out[-1]["t"] >= 1.5:
            out.append(p)
    plan["pauses"] = out
    return plan


def caption_at(t, plan, sc):
    if t < INTRO_S:
        return sc["intro"]
    ev = [e for e in plan["events"] if e[0] <= t < e[0] + EVENT_S and e[0] > 0.2]
    if ev:
        last = max(ev, key=lambda e: e[0])
        grp = [e[1] for e in ev if e[2] == last[2] and abs(e[0] - last[0]) <= 0.6]
        who = cows_txt(grp)
        zone = "feed bunk" if last[2] in ("reach_feed", "eat") else "water trough"
        secs = "3" if zone == "feed bunk" else "2"
        verbs = {"reach_feed": f"{who} reached the feed bunk. A {secs}-second timer starts (yellow CHECKING label).",
                 "reach_water": f"{who} reached the water trough. A {secs}-second timer starts (yellow CHECKING label).",
                 "eat": f"{who} stood at the feed bunk for {secs} seconds, so the label becomes EATING.",
                 "drink": f"{who} stood at the water trough for {secs} seconds, so the label becomes DRINKING.",
                 "leave": f"{who} moved away from the zone and is back to STANDING."}
        return verbs[last[2]]
    return "Each cow has its own record on the right: ID, behaviour, zone and confidence. The timeline logs every cow over time."


# ---------------------------------------------------------------- drawing helpers

def font(size, bold=False):
    return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size, index=1 if bold else 0)


F = {k: font(*v) for k, v in dict(h1=(64, True), h2=(34, True), h3=(26, True), body=(22, False), cap=(24, False),
                                  small=(18, False), tiny=(16, False), big=(54, True), num=(48, True),
                                  label=(20, True), mono=(20, False), step=(30, True)).items()}
BG, PANEL, LINE, TXT, MUTED, ACC = (12, 22, 18), (20, 34, 28), (44, 66, 56), (234, 242, 237), (140, 163, 151), (110, 226, 150)
VX, VY, VW, VH = 24, 104, 1248, 702  # video box on the canvas


def fmt_t(t):
    t = max(0.0, t)
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


def rounded(d, box, r, fill=None, outline=None, width=1):
    d.rounded_rectangle(box, r, fill=fill, outline=outline, width=width)


def wrap(d, text, fnt, maxw):
    lines, cur = [], ""
    for word in text.split():
        trial = (cur + " " + word).strip()
        if d.textlength(trial, font=fnt) <= maxw:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def dashed_poly(d, pts, color, width=3, dash=14):
    for i in range(len(pts)):
        (x1, y1), (x2, y2) = pts[i], pts[(i + 1) % len(pts)]
        L = max(1.0, float(np.hypot(x2 - x1, y2 - y1)))
        n = int(L // dash)
        for k in range(0, n, 2):
            a, b = k / n, min(1.0, (k + 1) / n)
            d.line([(x1 + (x2 - x1) * a, y1 + (y2 - y1) * a), (x1 + (x2 - x1) * b, y1 + (y2 - y1) * b)], fill=color, width=width)


def header(d, scene_title, idx, n_scenes):
    d.rectangle([0, 0, W, 84], fill=PANEL)
    d.line([0, 84, W, 84], fill=LINE, width=2)
    d.ellipse([28, 30, 52, 54], fill=ACC)
    d.text((66, 20), "FarmPulse AI", font=F["h2"], fill=TXT)
    d.text((300, 30), "Vision module  ·  cow detection and behaviour", font=F["body"], fill=MUTED)
    tag = f"Scene {idx}/{n_scenes}  ·  {scene_title}"
    tw = d.textlength(tag, font=F["h3"])
    rounded(d, (W - 68 - tw, 18, W - 28, 66), 24, fill=BG, outline=LINE, width=2)
    d.text((W - 48 - tw, 27), tag, font=F["h3"], fill=TXT)


def nearest(box, p):
    return (min(max(p[0], box[0]), box[2]), min(max(p[1], box[1]), box[3]))


def draw_bubble(d, text, anchor, color):
    """Callout bubble next to `anchor` (x1,y1,x2,y2 in canvas px), kept inside the video area.
    anchor=None puts it top-centre with no pointer."""
    lines = wrap(d, text, F["cap"], 470 if anchor else 760)
    w = max(d.textlength(l, font=F["cap"]) for l in lines) + 40
    h = len(lines) * 31 + 30
    if anchor is None:
        bx, by = VX + (VW - w) / 2, VY + 24
        rounded(d, (bx, by, bx + w, by + h), 14, fill=(12, 22, 18, 240), outline=color + (255,), width=3)
        for i, l in enumerate(lines):
            d.text((bx + 20, by + 14 + i * 31), l, font=F["cap"], fill=TXT)
        return
    x1, y1, x2, y2 = anchor
    cands = [(x2 + 26, y1), (x1 - 26 - w, y1), (x1, y2 + 26), (x1, y1 - 26 - h), (VX + 30, VY + 30)]
    pos = cands[-1]
    for cx, cy in cands:
        if VX + 8 <= cx and cx + w <= VX + VW - 8 and VY + 8 <= cy and cy + h <= VY + VH - 8:
            pos = (cx, cy)
            break
    bx, by = pos
    bx = min(max(bx, VX + 8), VX + VW - w - 8)
    by = min(max(by, VY + 8), VY + VH - h - 8)
    bb = (bx, by, bx + w, by + h)
    pa = nearest(anchor, ((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2))
    pb = nearest(bb, pa)
    d.line([pa, pb], fill=color + (255,), width=4)
    d.ellipse([pa[0] - 7, pa[1] - 7, pa[0] + 7, pa[1] + 7], fill=color + (255,))
    rounded(d, bb, 14, fill=(12, 22, 18, 240), outline=color + (255,), width=3)
    for i, l in enumerate(lines):
        d.text((bx + 20, by + 14 + i * 31), l, font=F["cap"], fill=TXT)


def draw_scene_frame(canvas, sc, res, plan, src, t, n_scenes, sidx, pause=None):
    """canvas: PIL RGB 1920x1080 (background already filled). src: BGR source frame. t: scene time (s)."""
    fps, (w, h) = res["fps"], res["size"]
    f_idx = res["start_f"] + int(round(t * fps))
    vid = Image.fromarray(cv2.cvtColor(cv2.resize(src, (VW, VH), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
    sx, sy = VW / w, VH / h

    def tv(x, y):
        return (VX + x * sx, VY + y * sy)

    fr_f = res["_frame_ids"]
    j = bisect.bisect_right(fr_f, f_idx) - 1
    cur = res["frames"][max(j, 0)]
    dets = sorted([(dd, st) for dd, st in zip(cur["d"], cur["s"])], key=lambda z: z[0][2])
    target = None
    if pause:
        target = set(range(10 ** 6)) if pause["ids"] == "all" else set(pause["ids"])

    # dim the video around the cow being explained
    if pause and pause["ids"] != "all":
        dim = Image.new("RGB", (VW, VH), (0, 0, 0))
        dimmed = Image.blend(vid, dim, 0.55)
        for dd, st in dets:
            if dd[0] in target:
                bx1, by1, bx2, by2 = [int(v) for v in (dd[2] * sx, dd[3] * sy, dd[4] * sx, dd[5] * sy)]
                bx1, by1, bx2, by2 = max(bx1, 0), max(by1, 0), min(bx2, VW), min(by2, VH)
                if bx2 > bx1 and by2 > by1:
                    dimmed.paste(vid.crop((bx1, by1, bx2, by2)), (bx1, by1))
        vid = dimmed
    canvas.paste(vid, (VX, VY))

    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    s = res["shift"].get(str(min(max(f_idx, res["start_f"]), res["start_f"] + len(res["shift"]) - 1)), [0, 0])
    zcol = {"feed": ORANGE, "water": BLUE}
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

    counts = defaultdict(int)
    for dd, st in dets:
        counts[st[0]] += 1
    show = [(dd, st) for dd, st in dets if (not pause or pause["ids"] == "all" or dd[0] in target)]
    crowded = len(show) > 5
    for dd, st in show:
        c = STATE_COL[st[0]]
        bw = 6 if pause else 4
        d.rounded_rectangle([tv(*dd[2:4]), tv(*dd[4:6])], 8, outline=c + (255,), width=bw)
    placed = []
    for dd, st in show:
        tid, conf, x1, y1 = dd[:4]
        state, dwell = st
        c = STATE_COL[state]
        if state.startswith("pend_"):
            txt = f"Cow {tid}  Checking {min(dwell, NEED[state[5:]]):.1f}/{NEED[state[5:]]:.0f} s"
        else:
            txt = f"Cow {tid}  {STATE_NAME[state]}" + ("" if crowded else f"  {conf * 100:.0f}%")
        tw = d.textlength(txt, font=F["label"])
        p1 = tv(x1, y1)
        bx, by = p1[0], max(p1[1] - 34, VY + 4)
        bx = min(max(bx, VX + 4), VX + VW - tw - 22)
        for _ in range(8):  # nudge down until the label does not cover an earlier one
            if not any(bx < r[2] and bx + tw + 18 > r[0] and by < r[3] and by + 32 > r[1] for r in placed):
                break
            by += 34
        placed.append((bx, by, bx + tw + 18, by + 32))
        rounded(d, (bx, by, bx + tw + 18, by + 32), 8, fill=c + (245,))
        d.text((bx + 9, by + 5), txt, font=F["label"], fill=(14, 20, 16))
        if state.startswith("pend_"):  # timer bar under the label
            frac = min(1.0, dwell / NEED[state[5:]])
            d.rectangle([bx, by + 32, bx + tw + 18, by + 38], fill=(0, 0, 0, 160))
            d.rectangle([bx, by + 32, bx + (tw + 18) * frac, by + 38], fill=c + (255,))
    # pause callout
    if pause:
        if pause["ids"] == "all":
            draw_bubble(d, pause["text"], None, ACC)
        else:
            tg = [dd for dd, st in show if dd[0] in target]
            if tg:
                dd = tg[0]
                x1, y1 = tv(*dd[2:4])
                x2, y2 = tv(*dd[4:6])
                draw_bubble(d, pause["text"], (x1, y1, x2, y2), STATE_COL[dict((a[0], b) for a, b in dets)[dd[0]][0]])
    ov_c = ov.crop((VX, VY, VX + VW, VY + VH))
    canvas.paste(ov_c, (VX, VY), ov_c)
    d2 = ImageDraw.Draw(canvas, "RGBA")
    rounded(d2, (VX, VY, VX + VW, VY + VH), 2, outline=LINE, width=2)
    rounded(d2, (VX + 12, VY + VH - 46, VX + 232, VY + VH - 12), 8, fill=(0, 0, 0, 170))
    d2.ellipse([VX + 24, VY + VH - 36, VX + 38, VY + VH - 22], fill=(235, 70, 70) if not pause else YELLOW)
    d2.text((VX + 48, VY + VH - 42), f"CAM-{sidx:02d}  {fmt_t(t)}", font=F["small"], fill=(255, 255, 255))
    if pause:
        rounded(d2, (VX + VW - 168, VY + 12, VX + VW - 12, VY + 46), 8, fill=YELLOW + (240,))
        d2.text((VX + VW - 154, VY + 18), "PAUSED", font=F["label"], fill=(20, 20, 20))
    header(d2, sc["title"], sidx, n_scenes)

    # ---------------- right panel
    px, py, pw = 1296, 104, 600
    rounded(d2, (px, py, px + pw, py + 702), 14, fill=PANEL, outline=LINE, width=2)
    d2.text((px + 24, py + 16), "LIVE ANALYSIS", font=F["small"], fill=MUTED)
    n = str(len(cur["d"]))
    d2.text((px + 24, py + 42), n, font=F["big"], fill=TXT)
    d2.text((px + 24 + d2.textlength(n, font=F["big"]) + 14, py + 70), "cows detected now", font=F["body"], fill=MUTED)
    cw = (pw - 48 - 36) // 4
    for i, (key, name) in enumerate((("standing", "Standing"), ("pending", "Checking"), ("eating", "Eating"), ("drinking", "Drinking"))):
        cnt = counts["pend_eating"] + counts["pend_drinking"] if key == "pending" else counts[key]
        col = YELLOW if key == "pending" else STATE_COL[key]
        x0 = px + 24 + i * (cw + 12)
        rounded(d2, (x0, py + 124, x0 + cw, py + 212), 10, fill=BG, outline=col if cnt else LINE, width=2)
        d2.text((x0 + 14, py + 130), str(cnt), font=F["num"], fill=col if cnt else MUTED)
        d2.text((x0 + 14, py + 186), name, font=F["small"], fill=MUTED)

    d2.text((px + 24, py + 228), "COW RECORDS", font=F["small"], fill=MUTED)
    rows = sorted(dets, key=lambda z: z[0][0])[:7]
    for i, (dd, st) in enumerate(rows):
        y = py + 256 + i * 36
        state, dwell = st
        d2.ellipse([px + 26, y + 8, px + 40, y + 22], fill=STATE_COL[state])
        d2.text((px + 54, y + 2), f"Cow {dd[0]}", font=F["label"], fill=TXT)
        d2.text((px + 150, y + 2), STATE_NAME[state], font=F["label"], fill=STATE_COL[state])
        if state.startswith("pend_"):
            detail = f"{dwell:.1f} / {NEED[state[5:]]:.0f} s {ZONE_TXT[state]}"
        elif state in NEED:
            detail = f"{dwell:.1f} s {ZONE_TXT[state]}"
        else:
            detail = ZONE_TXT[state]
        d2.text((px + 280, y + 4), detail, font=F["small"], fill=MUTED)
        cs = f"{dd[1]:.2f}"
        d2.text((px + pw - 24 - d2.textlength(cs, font=F["small"]), y + 4), cs, font=F["small"], fill=MUTED)
    if not rows:
        d2.text((px + 26, py + 260), "no cows in view", font=F["body"], fill=MUTED)

    ey = py + 256 + 7 * 36 + 8
    d2.line([px + 24, ey - 6, px + pw - 24, ey - 6], fill=LINE, width=1)
    d2.text((px + 24, ey + 2), "EVENT LOG", font=F["small"], fill=MUTED)
    verbs = {"reach_feed": ("reached feed bunk", YELLOW), "eat": ("now EATING", ORANGE),
             "reach_water": ("reached water trough", YELLOW), "drink": ("now DRINKING", BLUE),
             "leave": ("left the zone, STANDING", GREEN)}
    evs = [e for e in plan["events"] if e[0] <= t][-3:]
    for i, (et, tid, typ) in enumerate(evs):
        y = ey + 28 + i * 26
        d2.text((px + 24, y), fmt_t(et), font=F["mono"], fill=MUTED)
        d2.text((px + 110, y), f"Cow {tid}", font=F["mono"], fill=TXT)
        d2.text((px + 200, y), verbs[typ][0], font=F["mono"], fill=verbs[typ][1])

    # ---------------- bottom-left timeline
    ty = 830
    rounded(d2, (24, ty, VX + VW, ty + 226), 14, fill=PANEL, outline=LINE, width=2)
    d2.text((48, ty + 14), "BEHAVIOUR TIMELINE PER COW", font=F["small"], fill=MUTED)
    lx = 600
    for key, name in (("standing", "Standing"), ("pend_eating", "Checking (timer)"), ("eating", "Eating"), ("drinking", "Drinking")):
        d2.rectangle([lx, ty + 18, lx + 18, ty + 36], fill=STATE_COL[key])
        d2.text((lx + 26, ty + 14), name, font=F["small"], fill=TXT)
        lx += 36 + d2.textlength(name, font=F["small"]) + 24
    ids = plan["tl_ids"][:6]
    x0, x1 = 150, VX + VW - 28
    dur = sc["dur"]
    for r, tid in enumerate(ids):
        y = ty + 48 + r * 23
        d2.text((48, y - 1), f"Cow {tid}", font=F["small"], fill=TXT)
        d2.rectangle([x0, y + 1, x1, y + 17], fill=BG)
        for (a, b, st) in plan["segments"][tid]:
            if a > t:
                break
            d2.rectangle([x0 + (x1 - x0) * a / dur, y + 1, x0 + (x1 - x0) * min(b, t) / dur, y + 17], fill=STATE_COL[st])
    xp = x0 + (x1 - x0) * min(t, dur) / dur
    d2.line([xp, ty + 44, xp, ty + 48 + max(len(ids), 1) * 23], fill=(255, 255, 255), width=2)
    d2.text((48, ty + 200), "YOLO11m detection  ·  BoT-SORT tracking + ID re-link  ·  rule-based behaviour from zone dwell time (not a trained model)",
            font=F["tiny"], fill=MUTED)

    # ---------------- bottom-right caption
    rounded(d2, (px, ty, px + pw, ty + 226), 14, fill=PANEL, outline=YELLOW if pause else LINE, width=2)
    d2.text((px + 24, ty + 14), "PAUSED  ·  EXPLAINER" if pause else "WHAT YOU ARE SEEING", font=F["small"], fill=YELLOW if pause else MUTED)
    text = pause["text"] if pause else caption_at(t, plan, sc)
    lines = wrap(d2, text, font(23), pw - 48)[:6]
    for i, l in enumerate(lines):
        d2.text((px + 24, ty + 46 + i * 29), l, font=font(23), fill=TXT)


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


def card_how():
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im, "RGBA")
    d.text((110, 70), "How a cow's behaviour is worked out", font=font(60, True), fill=TXT)
    d.text((110, 150), "Four steps, repeated on every frame of the camera footage", font=font(30), fill=MUTED)
    steps = [("1", "Detect", "YOLO11 finds every cow in the frame and gives each one a box and a confidence score."),
             ("2", "Track", "BoT-SORT follows each box over time, so a cow keeps one ID (Cow 1, Cow 2 ...). A repair step joins broken tracks."),
             ("3", "Zones", "Zones are drawn once per camera: the pen, the feed bunk, the water trough. Only cows inside the pen count."),
             ("4", "Rules", "A cow that stands at a trough long enough gets that behaviour. A cow anywhere else in the pen is standing.")]
    x = 110
    for n, name, desc in steps:
        rounded(d, (x, 230, x + 410, 570), 18, fill=PANEL, outline=LINE, width=2)
        d.ellipse([x + 24, 252, x + 80, 308], fill=ACC)
        d.text((x + 44, 256), n, font=F["step"], fill=(14, 20, 16))
        d.text((x + 96, 262), name, font=F["h2"], fill=TXT)
        for i, l in enumerate(wrap(d, desc, F["body"], 360)):
            d.text((x + 28, 336 + i * 32), l, font=F["body"], fill=MUTED)
        x += 435
    rules = [(GREEN, "STANDING", "inside the pen, not at a trough"),
             (YELLOW, "CHECKING", "just reached a trough: the timer is running (shown as a bar under the label)"),
             (ORANGE, "EATING", "standing at the feed bunk for 3 seconds or more"),
             (BLUE, "DRINKING", "standing at the water trough for 2 seconds or more")]
    y = 620
    for col, name, desc in rules:
        d.rounded_rectangle([110, y, 330, y + 54], 12, fill=col + (255,))
        d.text((130, y + 12), name, font=F["h3"], fill=(14, 20, 16))
        d.text((360, y + 14), desc, font=font(28), fill=TXT)
        y += 70
    d.text((110, 925), "Eating and drinking mean 'standing at the trough': the camera cannot see swallowing. These are rules, not a trained behaviour model.", font=F["small"], fill=MUTED)
    return im


def card_outro(groups):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im, "RGBA")
    d.text((120, 70), "What the system recorded", font=font(64, True), fill=TXT)
    d.text((120, 160), "Cow-seconds per behaviour (all cows and scenes of each place combined)", font=font(30), fill=MUTED)
    y = 250
    names = (("standing", "Standing", GREEN), ("eating", "Eating", ORANGE), ("drinking", "Drinking", BLUE))
    for title, secs in groups:
        tot = sum(secs.values()) or 1
        rounded(d, (120, y, W - 120, y + 140), 16, fill=PANEL, outline=LINE, width=2)
        d.text((150, y + 24), title, font=F["h2"], fill=TXT)
        d.text((150, y + 74), f"{tot:.0f} cow-seconds observed", font=F["body"], fill=MUTED)
        x0, x1 = 640, W - 160
        xx = x0
        for key, name, col in names:
            v = secs.get(key, 0)
            if v > 0:
                wv = (x1 - x0) * v / tot
                d.rectangle([xx, y + 36, xx + wv - 3, y + 100], fill=col)
                if wv > 110:
                    d.text((xx + 14, y + 54), f"{name} {100 * v / tot:.0f}%", font=F["label"], fill=(14, 20, 16))
                xx += wv
        y += 165
    rounded(d, (120, y + 10, W - 120, y + 270), 16, fill=PANEL, outline=LINE, width=2)
    d.text((150, y + 28), "Limits today", font=F["h3"], fill=YELLOW)
    d.text((150, y + 72), "Lying is not detected yet (no trained posture model)  ·  labels are rules, not a trained model  ·  crowded lines are hard for the detector", font=F["body"], fill=TXT)
    d.text((150, y + 120), "Next steps", font=F["h3"], fill=ACC)
    d.text((150, y + 164), "Fine-tune the detector on farm footage  ·  train a behaviour classifier  ·  cow re-identification  ·  suspicion model and alerts", font=F["body"], fill=TXT)
    return im


# ---------------------------------------------------------------- render

class Writer:
    def __init__(self, path):
        import imageio_ffmpeg
        self.p = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                   "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                                   "-crf", "20", "-preset", "medium", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)

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


def hold_card(wr, im, secs):
    FADE = int(0.35 * FPS)
    n = int(secs * FPS)
    for k in range(n):
        wr.write(fade(im, min(k / FADE, (n - k) / FADE)))


def render(out=OUT):
    wr = Writer(out)
    groups = defaultdict(lambda: defaultdict(float))
    FADE = int(0.35 * FPS)
    hold_card(wr, card_intro(), 4.0)
    hold_card(wr, card_how(), 12.0)
    for si, sc in enumerate(SCENES, 1):
        res = load_res(sc)
        plan = scene_plan(res, sc)
        for fr in res["frames"]:
            for l in fr["b"]:
                groups[sc["group"]][l] += res["stride"] / res["fps"]
        cap = cv2.VideoCapture(str(CD / sc["video"]))
        fps = res["fps"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, res["start_f"])
        nxt, last = res["start_f"], None
        n = int(sc["dur"] * FPS)
        pend = list(plan["pauses"])
        total = n + sum(int(HOLD_S * FPS) for _ in pend)
        written = 0
        for k in range(n):
            t = k / FPS
            while pend and t >= pend[0]["t"]:
                p = pend.pop(0)
                pf = res["start_f"] + int(round(p["t"] * fps))
                while nxt <= pf:
                    ok, fr_ = cap.read()
                    if ok:
                        last = fr_
                    nxt += 1
                canvas = Image.new("RGB", (W, H), BG)
                draw_scene_frame(canvas, sc, res, plan, last, p["t"], len(SCENES), si, pause=p)
                for _ in range(int(HOLD_S * FPS)):
                    wr.write(canvas)
                    written += 1
            target = res["start_f"] + int(round(t * fps))
            while nxt <= target:
                ok, fr_ = cap.read()
                if ok:
                    last = fr_
                nxt += 1
            canvas = Image.new("RGB", (W, H), BG)
            draw_scene_frame(canvas, sc, res, plan, last, t, len(SCENES), si)
            wr.write(fade(canvas, min((k + 1) / FADE, (n - k) / FADE)) if (k < FADE or n - k < FADE) else canvas)
            written += 1
        cap.release()
        print("rendered", sc["name"], "pauses:", [(round(p["t"], 1), p["ids"]) for p in plan["pauses"]], flush=True)
    hold_card(wr, card_outro(list(groups.items())), 9.0)
    wr.close()
    print("wrote", out)


def preview():
    for si, sc in enumerate(SCENES, 1):
        res = load_res(sc)
        plan = scene_plan(res, sc)
        cap = cv2.VideoCapture(str(CD / sc["video"]))

        def frame_at(t):
            cap.set(cv2.CAP_PROP_POS_FRAMES, res["start_f"] + int(round(t * res["fps"])))
            return cap.read()[1]
        jobs = [(f"t{int(f * 100)}", sc["dur"] * f, None) for f in (0.3, 0.75)]
        jobs += [(f"pause{i}", p["t"], p) for i, p in enumerate(plan["pauses"])]
        for tag, t, p in jobs:
            canvas = Image.new("RGB", (W, H), BG)
            draw_scene_frame(canvas, sc, res, plan, frame_at(t), t, len(SCENES), si, pause=p)
            canvas.save(WORK / f"preview_{sc['name']}_{tag}.jpg", quality=88)
        print(sc["name"], "events:", plan["events"][:12], "pauses:", [(round(p["t"], 1), p["ids"]) for p in plan["pauses"]])


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
