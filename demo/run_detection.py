"""Step 1 of the demo: run the repo's real detector + tracker (cow-detection/) on the demo segments.

Writes demo/work/<scene>.json with per-frame detections [id, conf, x1, y1, x2, y2] after
BoT-SORT tracking and the repo's offline ID re-linking. Behaviours are added in make_demo.py.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CD = ROOT / "cow-detection"
sys.path.insert(0, str(CD))

import video_processor as vp  # noqa: E402
from scenes import SCENES      # noqa: E402

WORK = Path(__file__).resolve().parent / "work"
WORK.mkdir(exist_ok=True)
vp.OUTPUT_DIR = WORK / "runs"   # keep demo runs out of cow-detection/outputs


def main(names):
    for sc in SCENES:
        if names and sc["name"] not in names:
            continue
        s = vp.ProcessingSettings(conf=0.35, imgsz=960, stride=sc["stride"], start_seconds=sc["start"],
                                  max_seconds=sc["dur"], behaviours=False, export_video=False)
        job = vp.VideoJob(str(CD / sc["video"]), s)
        job.run()
        print(sc["name"], job.status, job.error, "frames", len(job.frames),
              "unique ids", len({d[0] for f in job.frames for d in f["d"]}), flush=True)
        (WORK / f"{sc['name']}.json").write_text(json.dumps(
            {"fps": job.meta["fps"], "size": [job.meta["width"], job.meta["height"]],
             "stride": sc["stride"], "frames": job.frames}))


if __name__ == "__main__":
    main(set(sys.argv[1:]))
