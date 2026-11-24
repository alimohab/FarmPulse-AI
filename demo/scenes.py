"""Scene definitions for the demo.

Zone polygons are in SOURCE-video pixels at time `ref` (seconds into the video, default = start); make_demo.py
moves them with the measured camera motion because the clips are handheld. They were drawn by hand for this
demo (the repo's own zone editor does the same job for fixed cameras).
  pen   - only cows whose feet are inside are used (drops cows in neighbouring pens)
  feed  - standing at the feed bunk  -> "eating"   (after 3 s)
  water - standing at the water trough -> "drinking" (after 2 s)

pauses: which explainer freeze-frames to add - "detect" (boxes + IDs), "eat", "drink", "stand".
"""

BIG_PEN = lambda y1, y2: [(-6000, y1), (7000, y2), (7000, 1200), (-6000, 1200)]  # noqa: E731

SCENES = [
    dict(name="bunk", group="Feeding bunk", title="Feeding bunk", video="WhatsApp Video 2026-09-29 at 20.49.57 (2).mp4",
         start=8.0, dur=10.5, stride=1, min_h=0.10, pauses=["detect", "eat"],
         intro="Scene 1 · Feeding bunk. A handheld camera walks along a pen. YOLO11 finds each cow and the tracker gives it an ID. Cows in the neighbouring pens are ignored.",
         zones=dict(
             pen=BIG_PEN(320, 300),
             feed=[(0, 576), (0, 545), (420, 470), (760, 400), (1024, 330), (1024, 440), (850, 500), (700, 576)])),
    dict(name="water", group="Water trough", title="Water trough", video="WhatsApp Video 2026-09-29 at 20.49.57 (3).mp4",
         start=0.0, dur=10.0, stride=1, min_h=0.10, pauses=["drink", "stand"],
         intro="Scene 2 · Water trough. The blue zone marks the trough. A cow standing at it for 2 seconds is labelled DRINKING; cows elsewhere in the pen are STANDING.",
         zones=dict(
             pen=BIG_PEN(250, 250),
             water=[(300, 300), (615, 255), (835, 500), (835, 576), (300, 576)])),
    dict(name="water2", group="Water trough", title="Water trough (2)", video="WhatsApp Video 2026-09-29 at 20.49.57 (3).mp4",
         start=11.0, dur=10.0, stride=1, min_h=0.10, pauses=["drink"],
         intro="Scene 3 · The second half of the same clip. The camera has moved to the other side of the trough, so the zone is re-anchored and follows the camera again.",
         zones=dict(
             pen=BIG_PEN(250, 250),
             water=[(480, 330), (600, 340), (745, 355), (870, 330), (900, 480), (800, 576), (430, 576), (430, 470)])),
    dict(name="line", group="Feed line", title="Feed line", video="videos/from_zip_8.55.23/WhatsApp Video 2026-09-29 at 8.54.34 PM (1).mp4",
         start=17.0, dur=23.0, stride=2, min_h=0.10, zone_frac=0.08, pauses=["eat", "stand"],
         intro="Scene 4 · A crowded feed line, 23 seconds from a steady camera. Many cows eat side by side; cows away from the bunk stay STANDING.",
         zones=dict(
             feed=[(190, 478), (290, 385), (400, 305), (560, 230), (700, 200), (780, 195), (780, 478)])),
]
