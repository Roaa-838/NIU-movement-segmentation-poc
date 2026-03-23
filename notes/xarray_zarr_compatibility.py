# Test 6: xr.open_zarr fails on OCTRON's zarr v3 store — zarr.open() required.

import zarr, numpy as np, xarray as xr, tempfile, os

with tempfile.TemporaryDirectory() as tmpdir:
    zarr_path = os.path.join(tmpdir, "test.zarr")

    # Simulate OCTRON's zarr structure: keys like "1_masks", "2_masks"
    store = zarr.open(zarr_path, mode="w")
    store["1_masks"] = np.zeros((100, 480, 640), dtype=bool)
    store["2_masks"] = np.zeros((100, 480, 640), dtype=bool)

    # Test 1: can xr.open_zarr read a raw zarr store?
    try:
        ds = xr.open_zarr(zarr_path)
        print(f"PASS: xr.open_zarr works — vars: {list(ds.data_vars)}")
    except Exception as e:
        print(f"INFO: xr.open_zarr failed ({e}) — must use zarr API directly")
        # Test 2: raw zarr API (always works)
        store2 = zarr.open(zarr_path, mode="r")
        mask = store2["1_masks"][0]   # single frame, lazy
        print(f"PASS: zarr direct API works — frame shape: {mask.shape}")
        print("Proposal note: use zarr.open() not xr.open_zarr() for OCTRON masks")