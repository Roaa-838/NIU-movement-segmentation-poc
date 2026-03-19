# Issue #301 Key Notes — Segmentation Tracking

My reading of the full thread. I've pulled out the comments that actually
matter for the proposal — who said what, what position they hold, and what
it implies for the technical approach. Worth re-reading the thread yourself
before submitting, but this should be enough to write from.

---

## Who said what (critical for proposal)

### @sfmig (Sep 11, 2024) — opened the issue

- Proposed RLE (Run-Length Encoding) as the mask storage format, referencing
  COCO's `pycocotools` wrappers (simplified BSD license, so licensing is clean).
- Asked whether we could link masks to bounding boxes:
  *"Maybe we can consider to link masks to bounding boxes too?"*
- **What this means for me:** Yes, link them. The dataset should have both
  centroids (position) and bboxes (shape). This directly answers sfmig's
  question and makes the segmentation dataset a superset of the bboxes one.

---

### @niksirbi (Nov 5, 2024)

- *"Given the growing popularity of models like SAM, I think the appetite for
  this sort of thing will keep growing, which means we should probably implement
  this."*
- **What this means for me:** Must-have feature in the eyes of the core team,
  not a speculative one. Frame it that way in the proposal — don't hedge.

---

### @gchindemi (Nov 11, 2024) — LISBET developer

- *"We want to experiment with segmentation masks as input for social behavior
  analysis in LISBET."*
- **What this means for me:** Real downstream user waiting. Mention LISBET
  explicitly in the proposal — it shows the feature has traction beyond
  movement itself.

---

### @SkepticRaven (Nov 15, 2024) — most important technical input

- Flagged that RLE storage is more complicated than it looks:
  *"The data storage problem may end up being a bit more complex than using RLE."*
- Said they ended up storing **contours** instead of masks in their own work.
- Shared an open field dataset with segmentation predictions (get the URL from
  the issue thread — haven't saved it yet).
- Has a JABS behavior classifier with readers for their custom format — worth
  reading their repo for inspiration on how they handled the format problem.
- **What this means for me:** Contours in xarray is validated by someone who
  actually tried RLE first and moved away from it. Good justification for the
  hybrid approach. Use SkepticRaven's open field dataset as test data.

---

### @niksirbi (Nov 15, 2024)

- *"Thanks a lot @SkepticRaven, this will be very useful."*
- **What this means for me:** Core team respects SkepticRaven's judgment.
  Following their recommendation (contours over raw RLE) is aligned with what
  the mentors will likely favor.

---

### @sfmig (Feb 6, 2025) — updated position

- Quoted SkepticRaven's point and linked to pycocotools again:
  https://github.com/cocodataset/cocoapi
- Walking back slightly from "pure contours" toward using COCO's RLE wrappers
  as at least an option.
- Also said on Zulip (Feb 13): *"Re using SAM 2/3 with HuggingFace packages: I
  have used the transformers package and found it helpful. Maybe the first step
  could be migrating to sam2 with transformers, and then adding sam3 could be
  easier from there."*
- **What this means for me:** Two things. Don't drop RLE — the hybrid handles
  it by keeping RLE in `ds.attrs` for reconstruction. And use the HuggingFace
  `transformers` API for any SAM2/3 code, NOT the raw `segment-anything-2` repo.
  sfmig recommended this explicitly 5 weeks ago — using the raw repo would
  signal I didn't read the thread carefully.

---

### @niksirbi (Jan 7, 2026) — most recent and most important

- Introduced OCTRON: a new tool for animal segmentation and tracking with a
  napari GUI.
- Its developer `@horsto` has already expressed interest in a possible
  integration with movement.
- **What this means for me:** OCTRON is the primary target. Most recent signal
  from a mentor, specific, with an active developer ready to collaborate.
  Lead with OCTRON in the proposal.

---

## OCTRON ACTUAL OUTPUT FORMAT (critical — from Zulip Feb 13 + docs)

This is the most important correction to my earlier schema design.

Mikkel Roald-Arbøl (Zulip, Feb 13): *"There's essentially two 'keypoints' in
the data — the centroid of the segmented mask, and the bbox (technically two
points, x_min:y_min and x_max:y_max). As you already have a chosen way to
represent bboxes, I think that'll fit in well."*

Niko confirmed: *"Reading the data as tracks of bounding boxes should be
straightforward."*

Reading the actual docs confirmed exactly this. OCTRON outputs per-individual
`.csv` files with these columns:

```
frame_counter, frame_idx, track_id, label, confidence,
pos_x, pos_y,           ← centroid (or bbox center if non-detailed mode)
bbox_x_min, bbox_x_max, bbox_y_min, bbox_y_max,
bbox_area, bbox_aspect_ratio,
[+ optional scikit-image shape features if "Detailed" was selected]
```

The programmatic loading path is `YOLO_results` from the `octron` package:

```python
from octron import YOLO_results
yolo_results = YOLO_results(results_dir, verbose=True)
tracking = yolo_results.get_tracking_data(interpolate=False)
masks    = yolo_results.get_mask_data()  # zarr, only if needed
```

The prediction folder structure is:
```
video_folder/
└── octron_predictions/
    └── videoname_HybridSort/
        ├── label_track_id_1.csv   ← one CSV per individual
        ├── label_track_id_2.csv
        ├── prediction_metadata.json
        └── predictions.zarr       ← masks, only if you need them
```

**What this means for my schema:** OCTRON does NOT need a contours array. The
CSV already gives centroid and bbox directly. This fits into `ValidBboxesInputs`
shape with minimal changes — Niko literally said it's straightforward. Contours
are only needed for SAM2/YOLO loaders where I'm deriving shape from raw masks.

Correct framing for the proposal:
- **OCTRON loader** = quick win. CSVs → centroids + bboxes → existing bbox
  infrastructure. No new schema work required.
- **SAM2/YOLO loaders** = the harder problem. Compute centroids + bboxes from
  masks, optionally add contours as an extra variable.

This also makes the 12-week timeline much more believable to reviewers.

---

## NEW: OCTRON now supports SAM3 (March 6, 2026)

Horst posted on Bluesky (Mar 6, 2026 — 12 days ago): OCTRON now supports SAM3
models alongside SAM2, and also added YOLO26 support.

This happened after the Feb 13 Zulip meeting. Nobody in the channel has
mentioned it yet.

**What this means for me:** An OCTRON loader transitively handles SAM3-generated
tracking results by design, since OCTRON abstracts over the segmentation model
internally. Add a sentence to the proposal: *"OCTRON added SAM3 and YOLO26
support in March 2026, meaning an OCTRON loader will support SAM3-backed
tracking results without any additional work."* Mention this in the Zulip post
too — being the only applicant who noticed it is a genuine signal to mentors
that I'm following the ecosystem in real time, not just reading the issue.

---

## My technical position (what I'll write in the proposal)

1. **Primary format: OCTRON** — most recent mentor signal (Jan 2026), output is
   already centroid + bbox CSVs, fits existing bbox infrastructure, loader is
   the quick win. SAM3 and YOLO26 support added in Mar 2026 comes for free.

2. **Secondary format: SAM2** — most widely used segmentation+tracking tool,
   requires computing centroids + bboxes from raw masks, optionally adds
   contours as an extra variable.

3. **Storage: hybrid, contours optional**
   - `position (time, space, individuals)` — centroid x,y, same dims as bbox
     dataset. `compute_velocity` etc. work without changes.
   - `shape (time, space, individuals)` — bbox w,h, links to existing bbox
     visualization pipeline.
   - `confidence (time, individuals)` — tracking score.
   - `contours (time, individuals, contour_points, space)` — **optional**, only
     present for SAM2/YOLO where raw masks are available. NaN-padded, follows
     movement's missing-keypoint convention. Not present in OCTRON datasets.
   - RLE in `ds.attrs["rle_masks"]` — optional, for mask reconstruction.

4. **SAM2/3 API:** HuggingFace `transformers`, not the raw Facebook repo.
   sfmig recommended this explicitly on Zulip. Removes the `segment-anything-2`
   install dependency entirely.

5. **Test data:** SkepticRaven's open field dataset (save URL from issue thread).

---

## HuggingFace transformers vs raw SAM2 repo

sfmig (Zulip, Feb 13) explicitly recommended the `transformers` approach.
Any SAM2/3 code in the PoC should use this:

```python
# ❌ wrong — raw Facebook repo, not what sfmig recommended
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ✅ correct
from transformers import SamModel, SamProcessor

model     = SamModel.from_pretrained("facebook/sam2-hiera-large")
processor = SamProcessor.from_pretrained("facebook/sam2-hiera-large")
# note: gated model on HuggingFace, users need to request access
```
