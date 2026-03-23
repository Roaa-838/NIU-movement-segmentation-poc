# notes/schema_design.py
# Thinking through mask storage options before committing to a schema.
# The core problem: full binary masks are huge. I want to understand
# exactly how much is saved by each alternative so I can justify the
# hybrid approach in the proposal.
#
# NOTE: After reading the OCTRON docs and the Feb 13 Zulip thread, I now know
# that OCTRON's output is already centroid + bbox CSVs — no masks involved.
# The memory problem below only applies to SAM2/YOLO loaders where the raw
# output actually IS boolean mask arrays.

import numpy as np

n_frames      = 1000
n_individuals = 3
H, W          = 480, 640

print("=" * 60)
print("mask storage options — memory breakdown (SAM2/YOLO context)")
print("=" * 60)

# Option A: store the full boolean masks.
# Simplest possible approach — just keep everything SAM2 gives you.
# numpy uses 1 byte per bool (not 1 bit), so this adds up fast.
masks_bool  = np.zeros((n_frames, n_individuals, H, W), dtype=bool)
size_A_mb   = masks_bool.nbytes / 1e6
print(f"\nOption A — full binary masks")
print(f"  shape: {masks_bool.shape}")
print(f"  dtype: bool (1 byte per pixel in numpy, not 1 bit)")
print(f"  memory: {size_A_mb:.1f} MB for {n_frames} frames")
print(f"  at 30fps, 10 min video: ~{size_A_mb * 18:.0f} MB")
print(f"  verdict: impractical for any real experiment")

# Option B: RLE encoding.
# COCO uses this. Compresses runs of identical pixels, works well for
# sparse masks. But it's variable-length, which means it doesn't fit
# cleanly into numpy arrays or xarray.
rle_bytes_per_mask = 300  # rough estimate for a typical animal silhouette
size_B_mb          = (n_frames * n_individuals * rle_bytes_per_mask) / 1e6
print(f"\nOption B — RLE (estimated)")
print(f"  ~{rle_bytes_per_mask} bytes per mask")
print(f"  memory: {size_B_mb:.2f} MB")
print(f"  compression vs Option A: {size_A_mb / size_B_mb:.0f}x")
print(f"  problem: variable-length strings don't fit numpy arrays,")
print(f"           can't slice or index without decoding first")

# Option C: contour points, NaN-padded.
# This is what @SkepticRaven ended up doing after trying RLE. Store the
# outline of the mask rather than the mask itself. Fixed-length array
# (NaN where unused) means it fits in xarray natively.
max_contour_points = 500
contours           = np.full(
    (n_frames, n_individuals, max_contour_points, 2),
    np.nan,
    dtype=np.float32
)
size_C_mb = contours.nbytes / 1e6
print(f"\nOption C — contour points (NaN-padded to {max_contour_points} pts)")
print(f"  shape: {contours.shape}  (time, individuals, max_pts, space)")
print(f"  dtype: float32")
print(f"  memory: {size_C_mb:.1f} MB")
print(f"  compression vs Option A: {size_A_mb / size_C_mb:.1f}x")
print(f"  fits xarray natively, NaN=no point (same pattern movement uses")
print(f"  for missing pose keypoints), works with existing dim convention")
print(f"  downside: fixed max_pts means complex shapes get truncated")

# Hybrid — what I'm actually proposing, but with contours optional.
#
# For OCTRON: position + shape + confidence only. OCTRON already exports
# centroid + bbox as CSV — no masks, no contours needed. Fits directly
# into the existing bbox infrastructure.
#
# For SAM2/YOLO: same base, plus contours extracted from the masks.
# RLE tucked away in attrs for reconstruction if ever needed.
centroids       = np.zeros((n_frames, 2, n_individuals), dtype=np.float32)
bboxes_shape    = np.zeros((n_frames, 2, n_individuals), dtype=np.float32)
confidence      = np.zeros((n_frames, n_individuals),    dtype=np.float32)
contours_hybrid = np.full(
    (n_frames, n_individuals, max_contour_points, 2),
    np.nan,
    dtype=np.float32
)

base_mb  = (centroids.nbytes + bboxes_shape.nbytes + confidence.nbytes) / 1e6
total_mb = base_mb + contours_hybrid.nbytes / 1e6

print(f"\nhybrid proposal")
print(f"  OCTRON dataset (no contours):")
print(f"    position   (time, space, individuals): {centroids.shape}  float32")
print(f"    shape      (time, space, individuals): {bboxes_shape.shape}  float32")
print(f"    confidence (time, individuals):        {confidence.shape}     float32")
print(f"    memory: {base_mb:.1f} MB")
print(f"  SAM2/YOLO dataset (with optional contours):")
print(f"    + contours (time, individuals, pts, space): {contours_hybrid.shape}")
print(f"    memory: {total_mb:.1f} MB")
print(f"  RLE masks can live in ds.attrs for reconstruction — not default path")

print(f"\n{'='*60}")
print("proposed xarray schema")
print("="*60)
print("""
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
""")
print("why these dims:")
print("  position/shape use (time, space, individuals) — identical to the bboxes")
print("  dataset, so compute_velocity, filter_by_confidence, compute_path_length")
print("  all work on segmentation data with zero changes.")
print()
print("  OCTRON datasets are essentially bboxes datasets with ds_type='segmentation'.")
print("  The OCTRON loader is the quick win; SAM2/YOLO are the harder problem.")
print()
print("  contours follow movement's NaN convention for missing data (same as")
print("  missing pose keypoints). space dim keeps 'x','y' labels consistent.")
print("  Only included when the source format provides masks to extract from.")
