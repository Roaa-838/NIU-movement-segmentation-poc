# load_bboxes.py → load_segmentation.py Mapping

This is my working blueprint for building `load_segmentation.py`. The bboxes
loader is the closest thing we have to what segmentation needs — same dimension
conventions, same validator pattern, same file-to-numpy-to-dataset pipeline.
Rather than starting from scratch I'm treating it as a template and noting
exactly what carries over, what needs tweaking, and what's genuinely new.

**Key correction from Zulip (Feb 13):** OCTRON's output is already centroid +
bbox CSVs — it doesn't need the contours dimension at all. The split is:
- OCTRON loader → almost identical to the bboxes loader (quick win)
- SAM2/YOLO loaders → need the new contours array (harder problem)

---

## Direct reuse (copy, rename, adapt)

### `from_numpy()` → `from_numpy_segmentation()`

The bboxes version takes:

```
position_array:   (n_frames, n_space, n_individuals)  ← centroid x,y
shape_array:      (n_frames, n_space, n_individuals)  ← width, height
confidence_array: (n_frames, n_individuals)
```

The segmentation version takes the exact same three arrays (same shapes, same
semantics), plus one **optional** new one that's only populated for SAM2/YOLO:

```
contours_array: (n_frames, n_individuals, max_pts, 2)  ← NaN-padded outline
                                                          None for OCTRON
```

Internally it calls `ValidSegmentationInputs.to_dataset()`. Everything else —
error handling, the logger pattern, how `fps` is threaded through — stays the
same.

---

### `from_via_tracks_file()` → `from_octron_file()` and `from_sam2_file()`

Bboxes registers its loader like this:

```python
@register_loader("VIA-tracks", file_validators=[ValidVIATracksCSV])
```

Mine will register two separate loaders:

```python
@register_loader("OCTRON", file_validators=[ValidFile])
@register_loader("SAM2",   file_validators=[ValidFile])
```

The internal flow is the same pattern in both cases:

```
# bboxes
file → _numpy_arrays_from_via_tracks_file() → from_numpy()

# segmentation
file → _numpy_arrays_from_octron_file() → from_numpy_segmentation()
file → _numpy_arrays_from_sam2_file()   → from_numpy_segmentation()
```

---

### `_numpy_arrays_from_via_tracks_file()` → `_numpy_arrays_from_octron_file()`

**OCTRON is the simpler case.** The prediction folder contains one CSV per
individual (`label_track_id_N.csv`). Each CSV has `frame_idx`, `pos_x`,
`pos_y`, `bbox_x_min/x_max/y_min/y_max`, and `confidence` as columns —
already the exact values I need. I can load them with pandas, pivot on
`frame_idx`, and fill the numpy arrays directly. No mask computation needed.

The loader can either use `pandas.read_csv` directly or go through the `octron`
package's `YOLO_results` reader class (which also gives access to zarr masks if
needed later). I'll support both paths and document the dependency.

**SAM2 is the harder case.** Reads a pickle or JSON containing the
`video_segments` nested dict, then computes centroids + bboxes from the boolean
masks. This is where `_compute_centroids_from_masks()` and
`_compute_bboxes_from_masks()` come in.

---

## New additions (no bboxes equivalent)

Only needed for SAM2/YOLO loaders, not OCTRON:

- **`contours_array` extraction** using `cv2.findContours` — takes the boolean
  mask per frame per individual, finds the largest external contour, and packs
  it into the NaN-padded array.
- **`_compute_centroids_from_masks()`** — utility that takes a `(H, W)` bool
  mask and returns `(x, y)` centroid.
- **`_compute_bboxes_from_masks()`** — same but returns `(cx, cy, w, h)`.
- **`ValidSegmentationInputs`** — extends `_BaseDatasetInputs`, adds the
  optional `contours_array` field and overrides `to_dataset()` to include the
  contours `DataArray` when it's not None.

---

## Things to copy exactly (don't reinvent these)

```python
# Logger import
from movement.utils.logging import logger

# Attach source file path to dataset
ds.attrs["source_file"] = file_path.as_posix()

# Log on successful load
logger.info(f"Loaded segmentation data from {file_path}:\n{ds}")

# Error raising — use this pattern, not bare raise
raise logger.error(ValueError("..."))

# File path handling
file_path = ValidFile(file_path)
```

---

## New file structure

```
movement/io/load_segmentation.py       ← new file
    from_numpy_segmentation()
    from_octron_file()                 ← @register_loader("OCTRON", ...)
    from_sam2_file()                   ← @register_loader("SAM2", ...)
    _numpy_arrays_from_octron_file()   ← reads CSV directly, no mask math
    _numpy_arrays_from_sam2_file()     ← reads masks, computes centroids/bboxes
    _compute_centroids_from_masks()    ← SAM2/YOLO only
    _compute_bboxes_from_masks()       ← SAM2/YOLO only

movement/validators/datasets.py        ← add to existing file
    ValidSegmentationInputs            ← extends _BaseDatasetInputs
        DIM_NAMES  = ("time", "space", "individuals")   ← same as bboxes
        VAR_NAMES  = ("position", "shape", "confidence")  ← + "contours" if present
        contours_array: Optional[np.ndarray]
        to_dataset()   ← adds contours DataArray only when contours_array is not None
```
