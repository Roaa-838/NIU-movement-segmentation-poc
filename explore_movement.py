import movement.sample_data as sd
from movement.io import load_bboxes, load_poses
from movement.kinematics import compute_velocity

# Reference dataset: VIA crabs (position + shape + confidence)
print("Bboxes Dataset Reference")
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


# Poses dataset (different dims, contains 'keypoints')
print("Poses Dataset Reference")
pose_ds = sd.fetch_dataset("SLEAP_three-mice_Aeon_proofread.analysis.h5")
print(pose_ds)
print()

print("Data variables:")
for name, var in pose_ds.data_vars.items():
    print(f"  {name}: shape={var.shape}, dims={var.dims}, dtype={var.dtype}")


# Checking specific dimension ordering to match for the OCTRON loader

print("Target Dimension Conventions")
print(f"bbox position dims:   {bbox_ds['position'].dims}")
print(f"bbox shape dims:      {bbox_ds['shape'].dims}")
print(f"bbox confidence dims: {bbox_ds['confidence'].dims}")


# verify compute_velocity works on bbox position arrays
print("Testing compute_velocity on bbox position")
vel = compute_velocity(bbox_ds['position'])
print(f"Velocity output shape: {vel.shape}")
print(f"Velocity output dims:  {vel.dims}")