import movement.sample_data as sd
from movement.io import load_bboxes, load_poses

# The VIA crabs dataset is the closest thing to what I'm building.
# Bboxes already has position + shape + confidence with the (time, space,
# individuals) convention, so this is my primary reference.
print("=" * 60)
print("bboxes dataset — primary template")
print("=" * 60)
bbox_ds = sd.fetch_dataset("VIA_multiple-crabs_5-frames_labels.csv")
print(bbox_ds)
print()

print("Data variables:")
for name, var in bbox_ds.data_vars.items():
    print(f"  {name}: shape={var.shape}, dims={var.dims}, dtype={var.dtype}")
print()

print("Coords:")
for name, coord in bbox_ds.coords.items():
    print(f"  {name}: {coord.values[:3]}...")
print()

print("Attrs:", bbox_ds.attrs)

# Poses dataset for contrast — different dim convention (has 'keypoints'),
# useful to see what I'm NOT trying to match.
print("\n" + "=" * 60)
print("poses dataset — for contrast only")
print("=" * 60)
pose_ds = sd.fetch_dataset("SLEAP_three-mice_Aeon_proofread.analysis.h5")
print(pose_ds)
print()

print("Data variables:")
for name, var in pose_ds.data_vars.items():
    print(f"  {name}: shape={var.shape}, dims={var.dims}, dtype={var.dtype}")

# The dim ordering is the thing I most need to get right.
# Checking it explicitly rather than assuming from docs.
print("\n" + "=" * 60)
print("dim convention — what to copy exactly")
print("=" * 60)
print("bbox position dims:   ", bbox_ds['position'].dims)
print("bbox shape dims:      ", bbox_ds['shape'].dims)
print("bbox confidence dims: ", bbox_ds['confidence'].dims)
# position and shape → ('time', 'space', 'individuals')
# confidence → ('time', 'individuals')  — no space dim, which makes sense

print()
print("confidence has no 'space' dim — it's a scalar per individual per frame.")
print("my segmentation confidence array will be the same shape.")

# Quick sanity check that compute_velocity actually works on this data.
# If it works on bbox centroids, it'll work on my mask centroids too since
# they'll have identical dims.
print("\n" + "=" * 60)
print("does compute_velocity work on bbox position as-is?")
print("=" * 60)
from movement.kinematics import compute_velocity
vel = compute_velocity(bbox_ds['position'])
print(f"output shape: {vel.shape}")
print(f"output dims:  {vel.dims}")
print("yes — same call will work on segmentation centroids with identical dims")