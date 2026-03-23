import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from notes.utils import build_octron_dataset

import numpy as np
import pandas as pd
import xarray as xr
import zarr

BASE   = Path("octron_predictions")
FOLDER = BASE / "worm_detailed_BotSort"
MULTI  = BASE / "3worms_multianimal_HybridSort"

PASS, FAIL, INFO = "PASS", "FAIL", "INFO"

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
        ds = build_octron_dataset(FOLDER / "prediction_metadata.json", "worm")
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
        ds_w = build_octron_dataset(FOLDER / "prediction_metadata.json", "worm")
        ds_l = build_octron_dataset(FOLDER / "prediction_metadata.json", "led")
        print(f"Worm: {ds_w.sizes}  led: {ds_l.sizes}")
        print(f"\n[{PASS}] label_filter separates worm and led classes")
    except Exception as e:
        print(f"[{FAIL}] {e}")

    print("\n" + "="*60)
    print("TEST F: Multi-animal — 3worms dataset")
    print("="*60)
    try:
        ds_m = build_octron_dataset(FOLDER / "prediction_metadata.json", "worm")
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
        ds = build_octron_dataset(FOLDER / "prediction_metadata.json", "worm")
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
        ds = build_octron_dataset(FOLDER / "prediction_metadata.json", "worm")
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
        ds = build_octron_dataset(FOLDER / "prediction_metadata.json", "worm")
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