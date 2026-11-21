# FarmPulse AI vision demo (mockup video)

`farmpulse_demo.mp4` (40 s, 1080p): cow detection plus standing / eating / drinking, on the project's own footage.

## How it is made

| Part | Source |
|---|---|
| Cow boxes, confidences, track IDs | Real output of `cow-detection/` (YOLO11m + BoT-SORT + ID re-link) |
| Standing / eating / drinking | `cow-detection/behaviour.py` rules: standing at a feed bunk >= 3 s = eating, at a water trough >= 2 s = drinking, otherwise standing |
| Trough and pen zones | Drawn by hand per scene in `scenes.py`, moved with the measured camera motion (the clips are handheld) |
| Dashboard panels, timeline, event log | Drawn by `make_demo.py` from the labelled detections |

Not a trained behaviour model, and eating / drinking mean "standing at the trough", as in the main README.
Lying is not shown (the demo footage has no usable lying cows and no posture model is trained yet).

## Rebuild

```bash
cd cow-detection && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && cd ../demo
../cow-detection/.venv/bin/python run_detection.py     # real detection + tracking -> work/*.json
../cow-detection/.venv/bin/python make_demo.py analyse  # behaviours
../cow-detection/.venv/bin/python make_demo.py render   # -> farmpulse_demo.mp4
```

Scenes (clip, start, length, zones) are defined in `scenes.py`.
