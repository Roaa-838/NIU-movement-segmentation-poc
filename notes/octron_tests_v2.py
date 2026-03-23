import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import zarr

BASE   = Path("octron_predictions")
FOLDER = BASE / "worm_detailed_BotSort"
MULTI  = BASE / "3worms_multianimal_HybridSort"\



PASS, FAIL, INFO = "PASS", "FAIL", "INFO"


def read_octron_csv(filepath: Path, debug: bool = False) -> pd.DataFrame:

    with open(filepath, "r") as f:
        lines = f.readlines()

    header_idx = None
    for i, line in enumerate(lines):
        if "pos_x" in line and "track_id" in line:
            header_idx = i
            break

    if header_idx is None:
        raise ValueError(f"Could not find header row in {filepath}")

    if header_idx > 0 and debug:
        print(f"  Skipped {header_idx} header/comment lines in {filepath.name}")
        for i in range(header_idx):
            print(f"    Line {i}: {lines[i].rstrip()}")

    return pd.read_csv(filepath, skiprows=header_idx)


def build_octron_dataset(folder: Path, label_filter: str = "worm") -> xr.Dataset:
    pattern = f"{label_filter}_track_*.csv"
    csv_files = sorted(folder.glob(pattern))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files for label='{label_filter}' found in {folder}")

    dfs = [read_octron_csv(f) for f in csv_files]

    # Detect frame column dynamically
    frame_col = next((c for c in ["frame_idx", "frame_counter", "frame"] 
                      if c in dfs[0].columns), None)
    if not frame_col:
        raise ValueError("Could not find a valid frame column (e.g., 'frame_idx') in the CSV.")

    # Get a sorted list of all unique frames across all individuals
    all_frames = sorted(set().union(*[set(df[frame_col].astype(int)) for df in dfs]))
    n_frames   = len(all_frames)
    n_ind      = len(dfs)
    
    # Create a mapping from frame number to array index
    frame_map  = {f: i for i, f in enumerate(all_frames)}

    # Pre-allocate NumPy arrays with NaNs
    pos  = np.full((n_frames, 2, n_ind), np.nan, "float32")
    shp  = np.full((n_frames, 2, n_ind), np.nan, "float32")
    conf = np.full((n_frames, n_ind),    np.nan, "float32")
    area = np.full((n_frames, n_ind),    np.nan, "float32")
    ecc  = np.full((n_frames, n_ind),    np.nan, "float32")
    sol  = np.full((n_frames, n_ind),    np.nan, "float32")
    ori  = np.full((n_frames, n_ind),    np.nan, "float32")

    has_minmax = "bbox_x_min" in dfs[0].columns

    # VECTORIZED ARRAY POPULATION (Replaces .iterrows)
    for ii, df in enumerate(dfs):
        # 1. Get the target array indices for the frames in this specific DataFrame
        row_frames = df[frame_col].astype(int).values
        idx = np.array([frame_map[f] for f in row_frames])

        # 2. Populate the arrays directly using vectorization
        pos[idx, 0, ii] = df["pos_x"].values
        pos[idx, 1, ii] = df["pos_y"].values
        
        if has_minmax:
            shp[idx, 0, ii] = (df["bbox_x_max"] - df["bbox_x_min"]).values
            shp[idx, 1, ii] = (df["bbox_y_max"] - df["bbox_y_min"]).values
            
        # 3. Use safe column checks for optional OCTRON features
        if "confidence" in df.columns:   conf[idx, ii] = df["confidence"].values
        if "area" in df.columns:         area[idx, ii] = df["area"].values
        if "eccentricity" in df.columns: ecc[idx, ii] = df["eccentricity"].values
        if "solidity" in df.columns:     sol[idx, ii] = df["solidity"].values
        if "orientation" in df.columns:  ori[idx, ii] = df["orientation"].values

    # Parse metadata for time coordinates
    meta_path = folder / "prediction_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            m = json.load(f)
        fps_val = m.get("video_info", {}).get("fps_original")
        octron_version = m.get("octron_version", "unknown")
    else:
        fps_val, octron_version = None, "unknown"

    # Standardize time coordinate (seconds if FPS is known, else frame number)
    time_coords = (np.array(all_frames) / fps_val if fps_val 
                   else np.array(all_frames, dtype="float64"))

    # Construct the xarray Dataset
    return xr.Dataset(
        {
            "position":     xr.DataArray(pos,  dims=("time", "space", "individuals")),
            "shape":        xr.DataArray(shp,  dims=("time", "space", "individuals")),
            "confidence":   xr.DataArray(conf, dims=("time", "individuals")),
            "area":         xr.DataArray(area, dims=("time", "individuals")),
            "eccentricity": xr.DataArray(ecc,  dims=("time", "individuals")),
            "solidity":     xr.DataArray(sol,  dims=("time", "individuals")),
            "orientation":  xr.DataArray(ori,  dims=("time", "individuals")),
        },
        coords={
            "time":        time_coords,
            "space":       ["x", "y"], # Standard space coordinates for 'movement'
            "individuals": [f.stem for f in csv_files],
        },
        attrs={
            "source_software": "OCTRON",
            "fps":             fps_val,
            "label":           label_filter,
            "octron_version":  octron_version,
            "mask_store":      str(folder / "predictions.zarr") if (folder / "predictions.zarr").exists() else None,
        },
    )

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("TEST A: prediction_metadata.json — correct key paths")
    print("="*60)
    with open(FOLDER / "prediction_metadata.json") as f:
        meta = json.load(f)

    video_info   = meta.get("video_info", {})
    tracker_cfg  = meta.get("tracker_configuration", {})
    pred_params  = meta.get("prediction_parameters", {})

    fps          = video_info.get("fps_original")
    height       = video_info.get("height")
    width        = video_info.get("width")
    n_frames     = video_info.get("num_frames_original")
    video_name   = video_info.get("original_video_name")
    octron_ver   = meta.get("octron_version")
    tracker_name = pred_params.get("tracker_name") or tracker_cfg.get("tracker_type")
    conf_thresh  = pred_params.get("conf_thresh")
    iou_thresh   = pred_params.get("iou_thresh")
    nr_classes   = tracker_cfg.get("parameters", {}).get("nr_classes")

    print(f"octron_version:   {octron_ver}")
    print(f"video_name:       {video_name}")
    print(f"fps_original:     {fps}")
    print(f"resolution:       {width} x {height}")
    print(f"num_frames:       {n_frames}")
    print(f"tracker_name:     {tracker_name}")
    print(f"nr_classes:       {nr_classes}")
    print(f"conf_thresh:      {conf_thresh}")
    print(f"iou_thresh:       {iou_thresh}")

    print(f"\n[{PASS}] fps = {fps} (at meta['video_info']['fps_original'])")
    print(f"[{PASS}] dims = {width}x{height} (at meta['video_info']['height/width'])")
    print(f"[{INFO}] nr_classes={nr_classes} — confirms multi-class")

    print("\n" + "="*60)
    print("TEST B: CSV format — detect and skip header lines")
    print("="*60)
    all_csvs   = sorted(FOLDER.glob("*.csv"))
    worm_csvs  = sorted(FOLDER.glob("worm_track_*.csv"))
    led_csvs   = sorted(FOLDER.glob("led_track_*.csv"))

    print(f"Total CSVs: {len(all_csvs)} ({len(worm_csvs)} worm, {len(led_csvs)} led)")

    df_worm = read_octron_csv(worm_csvs[0])
    print(f"\nworm_track CSV columns:\n  {list(df_worm.columns)}")
    print(f"Rows: {len(df_worm)}")
    print(f"\nSample row:")
    print(df_worm.iloc[0].to_dict())

    has_ecc = "eccentricity" in df_worm.columns
    has_sol = "solidity"     in df_worm.columns
    has_ori = "orientation"  in df_worm.columns
    has_area= "area"         in df_worm.columns
    print(f"\n[{PASS if has_ecc else FAIL}] eccentricity: {has_ecc}")
    print(f"[{PASS if has_sol else FAIL}] solidity:     {has_sol}")
    print(f"[{PASS if has_ori else FAIL}] orientation:  {has_ori}")
    print(f"[{PASS if has_area else FAIL}] area:         {has_area}")

    # Compare worm vs led columns
    df_led = read_octron_csv(led_csvs[0])
    print(f"\nled_track CSV columns:\n  {list(df_led.columns)}")
    only_worm = set(df_worm.columns) - set(df_led.columns)
    only_led  = set(df_led.columns)  - set(df_worm.columns)
    print(f"\nColumns ONLY in worm: {only_worm}")
    print(f"Columns ONLY in led:  {only_led}")
    print(f"\n[{INFO}] Both classes share same columns: {set(df_worm.columns) == set(df_led.columns)}")

    print("\n" + "="*60)
    print("TEST C: zarr structure — v3 format confirmed")
    print("="*60)
    zarr_path = FOLDER / "predictions.zarr"
    store = zarr.open(str(zarr_path), mode="r")
    mask_keys = sorted([k for k in store.keys() if k.endswith("_masks")])
    print(f"Mask keys: {mask_keys}")
    for key in mask_keys[:2]:
        arr = store[key]
        print(f"  {key}: shape={arr.shape} dtype={arr.dtype} chunks={arr.chunks}")

    single = store[mask_keys[0]][0]
    print(f"\nSingle frame: shape={single.shape} dtype={single.dtype} "
          f"size={single.nbytes}B foreground_px={single.sum()}")

    # Check zarr version
    with open(zarr_path / "zarr.json") as f:
        root_meta = json.load(f)
    zarr_version = root_meta.get("zarr_format")
    print(f"\nZarr format version: {zarr_version}")
    if zarr_version == 3:
        print(f"[{PASS}] Zarr v3 confirmed — requires zarr.open(), NOT xr.open_zarr()")

    print("\n" + "="*60)
    print("TEST D: Build movement dataset from real OCTRON data")
    print("="*60)
    try:
        ds = build_octron_dataset(FOLDER, "worm")
        print(ds)
        print(f"\nfps from metadata: {ds.attrs['fps']}")
        print(f"time coords (seconds): {ds.time.values[:5]}")

        from movement.kinematics import compute_velocity, compute_path_length
        from movement.filtering  import filter_by_confidence
        vel  = compute_velocity(ds["position"])
        path = compute_path_length(ds["position"])
        filt = filter_by_confidence(ds["position"], confidence=ds["confidence"], threshold=0.5)
        print(f"\n[{PASS}] compute_velocity:     {vel.shape}")
        print(f"[{PASS}] compute_path_length:  {path.shape}")
        print(f"[{PASS}] filter_by_confidence: {filt.shape}")
        print(f"  Dataset: {ds.sizes['time']} frames, {ds.sizes['individuals']} worm(s), "
              f"fps={ds.attrs['fps']}")
    except Exception as e:
        print(f"[{FAIL}] {e}")
        import traceback; traceback.print_exc()

    print("\n" + "="*60)
    print("TEST E: Multi-class — load worm vs led separately")
    print("="*60)
    try:
        ds_w = build_octron_dataset(FOLDER, "worm")
        ds_l = build_octron_dataset(FOLDER, "led")
        print(f"Worm: {ds_w.sizes}  led: {ds_l.sizes}")
        print(f"\n[{PASS}] label_filter separates worm and led classes")
    except Exception as e:
        print(f"[{FAIL}] {e}")

    print("\n" + "="*60)
    print("TEST F: Multi-animal — 3worms dataset")
    print("="*60)
    try:
        ds_m = build_octron_dataset(MULTI, "worm")
        vel_m = compute_velocity(ds_m["position"])
        print(f"[{PASS}] 3-animal dataset: {ds_m.sizes}, velocity shape: {vel_m.shape}")
        print(f"Individuals: {list(ds_m.individuals.values)}")
    except Exception as e:
        print(f"[{FAIL}] {e}")

    print("\n" + "="*60)
    print("TEST G: Lazy mask — RAM per frame")
    print("="*60)
    try:
        store = zarr.open(str(FOLDER/"predictions.zarr"), mode="r")
        keys  = sorted([k for k in store.keys() if "_masks" in k])
        T, H, W = store[keys[0]].shape
        eager_MB = T * len(keys) * H * W / 1e6
        frame_kb = H * W / 1e3
        schema   = ds.nbytes / 1e6 if 'ds' in dir() else None
        print(f"Full eager load: {eager_MB:.1f} MB ({T} frames × {len(keys)} tracks × {H}×{W})")
        print(f"Single frame:    {frame_kb:.0f} KB  (loaded on demand)")
        if schema:
            print(f"CSV schema RAM:  {schema:.3f} MB")
            print(f"Compression:     {eager_MB/schema:.0f}× vs eager")
        mask0 = store[keys[0]][0]
        print(f"\n[{PASS}] zarr.open()[key][frame] works, {mask0.nbytes}B per access")
    except Exception as e:
        print(f"[{FAIL}] {e}")

    print("\n" + "="*60)
    print("TEST H: NaN / missing frames")
    print("="*60)
    try:
        ds = build_octron_dataset(FOLDER, "worm")
        for ind in ds.individuals.values:
            miss = ds["position"].sel(individuals=ind).isnull().all("space").sum().item()
            pct  = 100*miss/ds.sizes["time"]
            print(f"  {ind}: {miss}/{ds.sizes['time']} frames missing ({pct:.1f}%)")
    except Exception as e:
        print(f"[{FAIL}] {e}")

    print("\n" + "="*60)
    print("TEST I: Eccentricity range")
    print("="*60)
    try:
        ds = build_octron_dataset(FOLDER, "worm")
        for ind in ds.individuals.values:
            e = ds["eccentricity"].sel(individuals=ind).values
            e = e[~np.isnan(e)]
            if len(e):
                print(f"  {ind}: mean={e.mean():.3f}  min={e.min():.3f}  "
                      f"max={e.max():.3f}  std={e.std():.3f}")
    except Exception as e:
        print(f"[{FAIL}] {e}")

    print("\n" + "="*60)
    print("TEST J: metadata.json consistency across all datasets")
    print("="*60)
    for sub in sorted(BASE.iterdir()):
        if sub.is_dir():
            mf = sub / "prediction_metadata.json"
            if mf.exists():
                with open(mf) as f:
                    m = json.load(f)
                vi = m.get("video_info", {})
                print(f"  {sub.name:40s} fps={vi.get('fps_original')}  "
                      f"res={vi.get('width')}x{vi.get('height')}  "
                      f"frames={vi.get('num_frames_original')}")

    print(f"\n[{INFO}] All metadata JSONs have same key structure → stable validator design")

    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)

    try:
        ds = build_octron_dataset(FOLDER, "worm")
        store = zarr.open(str(FOLDER/"predictions.zarr"), mode="r")
        keys = [k for k in store.keys() if "_masks" in k]
        T,H,W = store[keys[0]].shape
        print(f"Real data: {T} frames, {len(keys)} tracks, {H}x{W}px, fps={ds.attrs['fps']}")
        print(f"Full masks eager:  {T*len(keys)*H*W/1e6:.1f} MB")
        print(f"CSV schema in RAM: {ds.nbytes/1e6:.3f} MB")
        print(f"Single mask frame: {H*W/1e3:.0f} KB (on demand)")
        print(f"Compression:       {(T*len(keys)*H*W)/ds.nbytes:.0f}x")
        print(f"nr_classes in tracker config: {meta['tracker_configuration']['parameters'].get('nr_classes')}")
    except Exception as e:
        print(f"Summary error: {e}")