"""Scene definitions for the demo.

Zone polygons are in SOURCE-video pixels at the first frame of the segment; make_demo.py moves them
with the measured camera motion because the clips are handheld. They were drawn by hand for this
demo (the repo's own zone editor does the same job for fixed cameras).
  pen   - only cows whose feet are inside are used (drops cows in neighbouring pens)
  feed  - standing at the feed bunk  -> "eating"
  water - standing at the water trough -> "drinking"
"""

SCENES = [
    dict(name="bunk", title="Feeding bunk", video="WhatsApp Video 2026-09-29 at 20.49.57 (2).mp4",
         start=8.0, dur=12.0, stride=1, min_h=0.10,
         zones=dict(
             pen=[(-600, 320), (1624, 300), (1624, 1200), (-600, 1200)],
             feed=[(0, 576), (0, 545), (420, 470), (760, 400), (1024, 330), (1024, 440), (850, 500), (700, 576)])),
    dict(name="water", title="Water trough", video="WhatsApp Video 2026-09-29 at 20.49.57 (3).mp4",
         start=0.0, dur=10.0, stride=1, min_h=0.10,
         zones=dict(
             pen=[(-600, 250), (1624, 250), (1624, 1200), (-600, 1200)],
             water=[(300, 300), (615, 255), (835, 500), (835, 576), (300, 576)])),
    dict(name="line", title="Feed line", video="videos/from_zip_8.55.23/WhatsApp Video 2026-09-29 at 8.54.34 PM (1).mp4",
         start=17.0, dur=10.0, stride=2, min_h=0.10, zone_frac=0.08,
         zones=dict(
             feed=[(190, 478), (290, 385), (400, 305), (560, 230), (700, 200), (780, 195), (780, 478)])),
]
