# notes/sam2_format_study.py
# I want to understand the SAM2 output format before writing any loader code.
# No GPU needed here — just figuring out the data structure and what I'll
# actually be parsing.
#
# IMPORTANT: For any actual SAM2/3 inference, use HuggingFace transformers:
#
#   from transformers import SamModel, SamProcessor
#   model     = SamModel.from_pretrained("facebook/sam2-hiera-large")
#   processor = SamProcessor.from_pretrained("facebook/sam2-hiera-large")
#
# sfmig recommended this on Zulip (Feb 13) — it's simpler, already installed,
# and makes SAM3 migration easier later. Don't use the raw segment-anything-2
# repo. The output format (video_segments dict) is the same either way.
#
# This file is about the output format only, so no model imports needed here.

# SAM2 video tracking gives you back a nested dict that looks like this:
#
# video_segments = {
#     frame_idx: {
#         obj_id: {
#             "masks": np.ndarray,  # shape (1, H, W), dtype bool
#             "scores": float       # confidence, 0.0 to 1.0
#         }
#     }
# }
#
# frame_idx and obj_id are both ints. The mask has an extra leading dim
# (always 1) that I'll need to squeeze out. scores is just a plain float.

import numpy as np


def simulate_sam2_output(n_frames=10, n_individuals=2, H=480, W=640):
    """
    Fake SAM2 output for testing the loader without running the actual model.

    I'm using ellipse-shaped masks because they're a reasonable stand-in for
    animal silhouettes — simple enough to reason about, not so simple that
    they miss edge cases like mask overlap or partial frames.
    """
    rng = np.random.default_rng(42)
    video_segments = {}

    for frame_idx in range(n_frames):
        video_segments[frame_idx] = {}
        for obj_id in range(1, n_individuals + 1):
            # shift each individual horizontally so they don't overlap
            cy = H // 2 + int(rng.integers(-50, 50))
            cx = W // 2 + int(rng.integers(-50, 50)) + (obj_id - 1) * 100
            ry, rx = 60, 40
            y, x = np.ogrid[:H, :W]
            mask = ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2 <= 1

            video_segments[frame_idx][obj_id] = {
                "masks": mask[np.newaxis, :, :],  # keep the (1, H, W) shape SAM2 uses
                "scores": float(rng.uniform(0.7, 0.99))
            }

    return video_segments


video_segments = simulate_sam2_output()

print("SAM2 video_segments structure:")
print(f"  frame indices (first 3): {list(video_segments.keys())[:3]}...")
print(f"  obj IDs in frame 0: {list(video_segments[0].keys())}")
print(f"  keys per object: {list(video_segments[0][1].keys())}")
print(f"  mask shape: {video_segments[0][1]['masks'].shape}")  # expect (1, H, W)
print(f"  mask dtype: {video_segments[0][1]['masks'].dtype}")  # expect bool
print(f"  score sample: {video_segments[0][1]['scores']}")

print("\nConverting to movement-compatible arrays:")

n_frames = len(video_segments)
obj_ids = sorted(video_segments[0].keys())
n_individuals = len(obj_ids)
H, W = video_segments[0][1]["masks"].shape[1:]

print(f"  n_frames={n_frames}, n_individuals={n_individuals}, H={H}, W={W}")

# Centroids — movement convention is (time, space, individuals) where
# space = ["x", "y"]. x comes first, so space[0] = mean of x_coords.
centroids = np.full((n_frames, 2, n_individuals), np.nan, dtype=np.float32)

for frame_idx, frame_data in video_segments.items():
    for ind_idx, obj_id in enumerate(obj_ids):
        mask = frame_data[obj_id]["masks"][0]  # squeeze (1,H,W) → (H,W)
        if mask.any():
            y_coords, x_coords = np.where(mask)
            centroids[frame_idx, 0, ind_idx] = x_coords.mean()
            centroids[frame_idx, 1, ind_idx] = y_coords.mean()

print(f"  centroids shape: {centroids.shape}")  # (10, 2, 2)
print(f"  dims: (time, space, individuals)")
print(f"  centroids[0, :, 0] = {centroids[0, :, 0]}")  # x,y of individual 0 at frame 0

# Bboxes — I'm splitting into position (centroid of box) and shape (w, h)
# to match the existing bboxes dataset convention exactly.
bboxes_pos   = np.full((n_frames, 2, n_individuals), np.nan, dtype=np.float32)
bboxes_shape = np.full((n_frames, 2, n_individuals), np.nan, dtype=np.float32)

for frame_idx, frame_data in video_segments.items():
    for ind_idx, obj_id in enumerate(obj_ids):
        mask = frame_data[obj_id]["masks"][0]
        if mask.any():
            y_coords, x_coords = np.where(mask)
            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()
            w = x_max - x_min
            h = y_max - y_min
            bboxes_pos[frame_idx, 0, ind_idx]   = x_min + w / 2
            bboxes_pos[frame_idx, 1, ind_idx]   = y_min + h / 2
            bboxes_shape[frame_idx, 0, ind_idx] = w
            bboxes_shape[frame_idx, 1, ind_idx] = h

print(f"  bboxes_pos shape:   {bboxes_pos.shape}")
print(f"  bboxes_shape shape: {bboxes_shape.shape}")

# Confidence — straightforward, just pull the score per frame per individual.
confidence = np.full((n_frames, n_individuals), np.nan, dtype=np.float32)
for frame_idx, frame_data in video_segments.items():
    for ind_idx, obj_id in enumerate(obj_ids):
        confidence[frame_idx, ind_idx] = frame_data[obj_id]["scores"]

print(f"  confidence shape: {confidence.shape}")

# Contours — requires opencv. I'm NaN-padding to a fixed length because
# xarray needs rectangular arrays. 200 points covers most animal outlines;
# complex shapes get truncated to the largest contour.
# This step is SAM2/YOLO-specific — OCTRON doesn't need it since its CSV
# output already contains the centroid and bbox coordinates directly.
try:
    import cv2
    max_contour_points = 200
    contours_arr = np.full(
        (n_frames, n_individuals, max_contour_points, 2),
        np.nan,
        dtype=np.float32
    )

    for frame_idx, frame_data in video_segments.items():
        for ind_idx, obj_id in enumerate(obj_ids):
            mask = frame_data[obj_id]["masks"][0].astype(np.uint8) * 255
            contour_list, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contour_list:
                largest = max(contour_list, key=cv2.contourArea)
                pts = largest[:, 0, :]  # (N, 2) array of x,y points
                n_pts = min(len(pts), max_contour_points)
                contours_arr[frame_idx, ind_idx, :n_pts, :] = pts[:n_pts].astype(np.float32)

    print(f"  contours shape: {contours_arr.shape}")
    print(f"  dims: (time, individuals, contour_points, space)")
    print("  opencv found, contours extracted")

except ImportError:
    print("  opencv not available — install with: pip install opencv-python")
    print("  contours_arr set to None for now")
    contours_arr = None
    max_contour_points = 0

print("\nAll arrays ready for xarray.Dataset assembly")
print("(contours_arr will be None for OCTRON datasets — that's expected)")
