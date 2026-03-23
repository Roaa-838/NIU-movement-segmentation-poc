"""
notes/build_xarray_dataset.py

Builds the xarray.Dataset in the shape expected by movement.
Validates compatibility with existing kinematics functions.

Supports two variants:
  - OCTRON: position + shape + confidence (CSV provides centroids/bboxes directly)
  - SAM2/YOLO: same base + optional contours (derived from masks)

Note: For any real SAM2/3 inference use HuggingFace transformers:
  from transformers import SamModel, SamProcessor
sfmig recommended this on Zulip (Feb 13) — avoids the heavy segment-anything-2
install and makes SAM3 migration easier later.
"""

import numpy as np
import xarray as xr


# ── Simulate SAM2 output (no GPU needed — this is about the data structure) ──

def simulate_sam2_output(n_frames=10, n_individuals=2, H=480, W=640):
    """Fake SAM2 video_segments dict for testing without running the model."""
    rng = np.random.default_rng(42)
    video_segments = {}
    for frame_idx in range(n_frames):
        video_segments[frame_idx] = {}
        for obj_id in range(1, n_individuals + 1):
            cy = H // 2 + int(rng.integers(-50, 50))
            cx = W // 2 + int(rng.integers(-50, 50)) + (obj_id - 1) * 100
            ry, rx = 60, 40
            y, x = np.ogrid[:H, :W]
            mask = ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2 <= 1
            video_segments[frame_idx][obj_id] = {
                "masks":  mask[np.newaxis, :, :],   # (1, H, W)
                "scores": float(rng.uniform(0.7, 0.99)),
            }
    return video_segments


video_segments = simulate_sam2_output()

# Pull arrays out of the dict — same conversion a real SAM2 loader would do.
n_frames     = len(video_segments)
obj_ids      = sorted(video_segments[0].keys())
n_individuals = len(obj_ids)
H, W         = video_segments[0][1]["masks"].shape[1:]

centroids    = np.full((n_frames, 2, n_individuals), np.nan, dtype=np.float32)
bboxes_shape = np.full((n_frames, 2, n_individuals), np.nan, dtype=np.float32)
confidence   = np.full((n_frames, n_individuals),    np.nan, dtype=np.float32)

for frame_idx, frame_data in video_segments.items():
    for ind_idx, obj_id in enumerate(obj_ids):
        mask = frame_data[obj_id]["masks"][0]
        if mask.any():
            y_coords, x_coords = np.where(mask)
            centroids[frame_idx, 0, ind_idx] = x_coords.mean()
            centroids[frame_idx, 1, ind_idx] = y_coords.mean()
            bboxes_shape[frame_idx, 0, ind_idx] = x_coords.max() - x_coords.min()
            bboxes_shape[frame_idx, 1, ind_idx] = y_coords.max() - y_coords.min()
        confidence[frame_idx, ind_idx] = frame_data[obj_id]["scores"]

try:
    import cv2
    max_contour_points = 200
    contours_arr = np.full(
        (n_frames, n_individuals, max_contour_points, 2), np.nan, dtype=np.float32
    )
    for frame_idx, frame_data in video_segments.items():
        for ind_idx, obj_id in enumerate(obj_ids):
            mask = frame_data[obj_id]["masks"][0].astype(np.uint8) * 255
            contour_list, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contour_list:
                pts = max(contour_list, key=cv2.contourArea)[:, 0, :]
                n_pts = min(len(pts), max_contour_points)
                contours_arr[frame_idx, ind_idx, :n_pts, :] = pts[:n_pts].astype(np.float32)
except ImportError:
    contours_arr = None
    max_contour_points = 0

individual_names = [f"id_{obj_id}" for obj_id in obj_ids]


# ── Dataset builder ───────────────────────────────────────────────────────────

def build_dataset(centroids, bboxes_shape, confidence, individual_names,
                  n_frames, H, W, contours_arr=None, max_contour_points=0,
                  source_software="SAM2"):
 
    ds = xr.Dataset(
        data_vars={
            "position": xr.DataArray(
                centroids, dims=("time", "space", "individuals")
            ),
            "shape": xr.DataArray(
                bboxes_shape, dims=("time", "space", "individuals")
            ),
            "confidence": xr.DataArray(
                confidence, dims=("time", "individuals")
            ),
            **(
                {
                    "contours": xr.DataArray(
                        contours_arr,
                        dims=("time", "individuals", "contour_points", "space"),
                    )
                }
                if contours_arr is not None
                else {}
            ),
        },
        coords={
            "time":        np.arange(n_frames, dtype=np.int64),
            "space":       ["x", "y"],
            "individuals": individual_names,
            **(
                {"contour_points": np.arange(max_contour_points)}
                if contours_arr is not None
                else {}
            ),
        },
        attrs={
            "source_software":     source_software,
            "ds_type":             "segmentation",
            "fps":                 None,
            "original_resolution": [H, W],
        },
    )
    return ds


if __name__ == "__main__":
    sam2_ds = build_dataset(
        centroids, bboxes_shape, confidence, individual_names,
        n_frames, H, W,
        contours_arr=contours_arr,
        max_contour_points=max_contour_points,
        source_software="SAM2",
    )
    octron_ds = build_dataset(
        centroids, bboxes_shape, confidence, individual_names,
        n_frames, H, W,
        contours_arr=None,
        source_software="OCTRON",
    )

    print("SAM2 Dataset (with contours) ")
    print(sam2_ds)
    print("\nOCTRON Dataset (no contours) ")
    print(octron_ds)

    print("\nCompatibility Checks ")
    from movement.kinematics import compute_velocity, compute_path_length
    from movement.filtering import filter_by_confidence

    vel      = compute_velocity(sam2_ds["position"])
    path_len = compute_path_length(sam2_ds["position"])
    filtered = filter_by_confidence(
        sam2_ds["position"], confidence=sam2_ds["confidence"], threshold=0.8
    )
    print(f"compute_velocity:     {vel.shape}")
    print(f"compute_path_length:  {path_len.shape}")
    print(f"filter_by_confidence: {filtered.shape}")

    print("\nMemory Comparison ")
    full_masks   = np.zeros((n_frames, n_individuals, H, W), dtype=bool)
    sam2_bytes   = sum(v.nbytes for v in sam2_ds.data_vars.values())
    octron_bytes = sum(v.nbytes for v in octron_ds.data_vars.values())
    print(f"Full binary masks:  {full_masks.nbytes / 1e6:.1f} MB")
    print(f"SAM2 dataset:       {sam2_bytes / 1e6:.2f} MB  (includes contours)")
    print(f"OCTRON dataset:     {octron_bytes / 1e6:.3f} MB  (position+shape+conf only)")