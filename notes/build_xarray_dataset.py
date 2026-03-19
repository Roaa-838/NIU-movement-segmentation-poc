# notes/build_xarray_dataset.py
# Assembling the xarray.Dataset in the exact shape movement expects.
# This is the pattern from_numpy_segmentation() will use internally —
# I'm working it out here first so I can verify compatibility before
# touching any of the actual movement source files.
#
# Two dataset variants:
#   - OCTRON: position + shape + confidence only (CSV already has centroids+bboxes)
#   - SAM2/YOLO: same base + optional contours (derived from raw masks)
#
# For any SAM2/3 inference code, use HuggingFace transformers:
#   from transformers import SamModel, SamProcessor
# NOT the raw segment-anything-2 repo. sfmig recommended this explicitly
# on Zulip (Feb 13) and it removes the segment-anything-2 install dependency.

import numpy as np
import xarray as xr
from notes.sam2_format_study import (
    obj_ids,
    centroids,
    bboxes_shape,
    confidence,
    contours_arr,
    n_frames,
    max_contour_points,
    H,
    W,
    n_individuals
)

# Use "id_{obj_id}" as the individual name — matches what movement does
# with its bboxes loader (string labels, not bare integers).
individual_names = [f"id_{obj_id}" for obj_id in obj_ids]


def build_dataset(centroids, bboxes_shape, confidence, individual_names,
                  n_frames, H, W, contours_arr=None, max_contour_points=0,
                  source_software="SAM2"):
    """
    Build a movement-compatible segmentation dataset.

    contours_arr is optional — pass None for OCTRON datasets where the source
    file already provides centroid + bbox directly (no masks to extract from).
    Pass a (n_frames, n_individuals, max_pts, 2) array for SAM2/YOLO datasets.
    """
    ds = xr.Dataset(
        data_vars={
            # position = centroid x,y of the segmentation mask.
            # Dims are identical to the bboxes dataset — this is intentional
            # so that compute_velocity and friends work without any changes.
            "position": xr.DataArray(
                centroids,
                dims=("time", "space", "individuals"),
            ),

            # shape = bounding box width and height.
            # Same dims as bboxes dataset.
            "shape": xr.DataArray(
                bboxes_shape,
                dims=("time", "space", "individuals"),
            ),

            # confidence = tracking score, one value per individual per frame.
            "confidence": xr.DataArray(
                confidence,
                dims=("time", "individuals"),
            ),

            # contours = mask outline as NaN-padded x,y sequences.
            # Only included for SAM2/YOLO where raw masks are available.
            # OCTRON datasets won't have this variable at all.
            **({"contours": xr.DataArray(
                contours_arr,
                dims=("time", "individuals", "contour_points", "space"),
            )} if contours_arr is not None else {}),
        },
        coords={
            "time":        np.arange(n_frames, dtype=np.int64),
            "space":       ["x", "y"],
            "individuals": individual_names,
            **({"contour_points": np.arange(max_contour_points)}
               if contours_arr is not None else {}),
        },
        attrs={
            "source_software":     source_software,
            "ds_type":             "segmentation",
            "fps":                 None,  # set by caller if known
            "original_resolution": [H, W],
            "max_contour_points":  max_contour_points if contours_arr is not None else 0,
        },
    )
    return ds


# SAM2 dataset — includes contours
sam2_ds = build_dataset(
    centroids, bboxes_shape, confidence, individual_names,
    n_frames, H, W,
    contours_arr=contours_arr,
    max_contour_points=max_contour_points,
    source_software="SAM2",
)

# OCTRON dataset — no contours (OCTRON CSVs already give centroids + bboxes)
octron_ds = build_dataset(
    centroids, bboxes_shape, confidence, individual_names,
    n_frames, H, W,
    contours_arr=None,
    source_software="OCTRON",
)

print("=" * 60)
print("SAM2 dataset (with contours)")
print("=" * 60)
print(sam2_ds)
print()
print("=" * 60)
print("OCTRON dataset (no contours — fits existing bbox infrastructure)")
print("=" * 60)
print(octron_ds)
print()

# Verify that the existing movement API actually works on both datasets.
# If any of these fail, the dim convention is wrong somewhere.
print("=" * 60)
print("compatibility checks (using SAM2 dataset)")
print("=" * 60)

from movement.kinematics import compute_velocity
velocity = compute_velocity(sam2_ds["position"])
print(f"compute_velocity:       shape={velocity.shape}, dims={velocity.dims}")

from movement.filtering import filter_by_confidence
filtered = filter_by_confidence(
    sam2_ds["position"], confidence=sam2_ds["confidence"], threshold=0.8
)
print(f"filter_by_confidence:   shape={filtered.shape}, dims={filtered.dims}")

from movement.kinematics import compute_path_length
path_len = compute_path_length(sam2_ds["position"])
print(f"compute_path_length:    shape={path_len.shape}, dims={path_len.dims}")

# Make sure label-based selection works.
ind_pos = sam2_ds["position"].sel(individuals="id_1")
print(f".sel(individuals='id_1'): shape={ind_pos.shape}")

# Time slicing sanity check.
first_5 = sam2_ds.sel(time=slice(0, 4))
print(f".sel(time=slice(0,4)):  time dim={first_5.sizes['time']}")

print()
print("=" * 60)
print("memory comparison")
print("=" * 60)

# Just to make the case for not storing full masks in the dataset.
full_masks   = np.zeros((n_frames, n_individuals, H, W), dtype=bool)
sam2_bytes   = sum(v.nbytes for v in sam2_ds.data_vars.values())
octron_bytes = sum(v.nbytes for v in octron_ds.data_vars.values())
print(f"Full binary masks:     {full_masks.nbytes / 1e6:.1f} MB")
print(f"SAM2 dataset:          {sam2_bytes / 1e6:.1f} MB  (includes contours)")
print(f"OCTRON dataset:        {octron_bytes / 1e6:.1f} MB  (position+shape+confidence only)")

sam2_ds.to_netcdf("sam2_movement_poc.nc")
print(f"\nSaved SAM2 dataset to sam2_movement_poc.nc")
