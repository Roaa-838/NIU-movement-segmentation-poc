# Movement Segmentation PoC
GSoC 2026 — NIU Movement: Segmentation Tracking ([issue #301](https://github.com/neuroinformatics-unit/movement/issues/301))

## Goal
Demonstrate that SAM2 segmentation output can be converted into a
movement-compatible `xarray.Dataset`, and that existing movement kinematics
functions work on it without modification.

## Status
- [x] movement installed and test suite passing (1126 passed, 100% coverage)
- [x] Memory analysis: full masks vs RLE vs contours (`notes/schema_design.py`)
- [x] SAM2 output format studied and simulated (`notes/sam2_format_study.py`)
- [x] xarray Dataset assembled and verified (`notes/build_xarray_dataset.py`)
- [x] Compatibility confirmed: `compute_velocity`, `filter_by_confidence`, `compute_path_length`
- [x] Loader mapping: `load_bboxes.py` → `load_segmentation.py` (`notes/loader_mapping.md`)
- [x] Issue #301 thread annotated with technical position (`notes/issue_301_notes.md`)
- [ ] OCTRON format study (Day 4)
- [ ] Skeleton PR to movement (Day 4)

## Proposed Dataset Schema

```
<xarray.Dataset>
Dimensions:
    time:           n_frames
    space:          2
    individuals:    n_individuals
    contour_points: max_contour_points   ← only present for SAM2/YOLO datasets

Coordinates:
  * time         (time)         float64   seconds (if fps known) else frame index
  * space        (space)        <U1       'x' 'y'
  * individuals  (individuals)  object    'id_0' 'id_1' ...

Data variables:
    position     (time, space, individuals)                  float32  ← centroid
    shape        (time, space, individuals)                  float32  ← bbox w,h
    confidence   (time, individuals)                         float32  ← tracking score
    contours     (time, individuals, contour_points, space)  float32  ← NaN-padded
                 ↑ optional — only present for SAM2/YOLO, not OCTRON

Attributes:
    source_software:     'SAM2' | 'OCTRON'
    source_file:         '/path/to/file'
    fps:                 30.0
    ds_type:             'segmentation'
    original_resolution: [H, W]
    rle_masks:           <optional, for mask reconstruction, SAM2/YOLO only>
```

`position` and `shape` use `(time, space, individuals)` — identical to the
existing bboxes dataset. This means `compute_velocity`, `filter_by_confidence`,
and `compute_path_length` all work on segmentation data without any changes.

OCTRON datasets skip the `contours` variable entirely since OCTRON already
exports centroid + bbox as CSV. The OCTRON loader is essentially the bboxes
loader with a different file parser — the hard schema problem only applies to
SAM2/YOLO where centroids and bboxes have to be derived from raw boolean masks.

## Memory Analysis (1000 frames, 3 individuals, 480×640)

```
Option A — full binary masks
  shape: (1000, 3, 480, 640)
  memory: 921.6 MB
  at 30fps, 10-min video: ~16,589 MB
  verdict: impractical for any real experiment

Option B — RLE (estimated)
  ~300 bytes per mask
  memory: 0.90 MB  (1024x compression)
  problem: variable-length, can't slice or index without decoding

Option C — contour points (NaN-padded to 500 pts)
  shape: (1000, 3, 500, 2)
  memory: 12.0 MB  (76.8x compression)
  fits xarray natively, NaN=no point (same convention movement uses
  for missing pose keypoints)

Hybrid proposal
  OCTRON dataset (position + shape + confidence only): 0.1 MB
  SAM2/YOLO dataset (+ contours):                    12.1 MB
  RLE stored in ds.attrs for reconstruction if needed — not default path
```

## Compatibility Proof

```python
from movement.kinematics import compute_velocity, compute_path_length
from movement.filtering import filter_by_confidence

compute_velocity(ds['position'])                              # ✓
filter_by_confidence(ds['position'], ds['confidence'], 0.8)  # ✓
compute_path_length(ds['position'])                           # ✓
ds['position'].sel(individuals='id_1')                        # ✓
ds.sel(time=slice(0, 4))                                      # ✓
```

All pass. Same dims as the bboxes dataset — no changes to existing functions needed.

## Key Technical Decisions

**OCTRON first, SAM2 second.** OCTRON's output is already centroid + bbox CSVs
(confirmed from docs and Zulip Feb 13). The OCTRON loader reuses existing bbox
infrastructure with minimal changes. SAM2/YOLO loaders are the harder problem
— they require computing centroids and bboxes from raw boolean masks, plus
optional contour extraction via opencv. OCTRON also added SAM3 and YOLO26
support in March 2026, meaning an OCTRON loader transitively supports
SAM3-generated tracking results.

**HuggingFace `transformers` for SAM2/3, not the raw Facebook repo.** sfmig
recommended this on Zulip (Feb 13). It removes the `segment-anything-2`
install dependency and makes SAM3 migration straightforward.

**Contours optional.** The `contours` variable is only added when the source
format provides raw masks to extract from. OCTRON datasets won't have it.
This keeps the schema clean and the OCTRON loader simple.

## Relates to
[neuroinformatics-unit/movement#301](https://github.com/neuroinformatics-unit/movement/issues/301)
