"""
notes/utils.py

Shared loader using the official OCTRON YOLO_results API.

Horst (OCTRON developer) recommended using YOLO_results instead of
reading CSVs manually — it handles header-skipping, frame alignment,
and feature extraction correctly.

Key API facts confirmed from OCTRON source (yolo_results.py):
  - YOLO_results() takes a folder path, NOT the JSON file path
  - get_tracking_data() returns:
      {track_id: {'label': str, 'data': df_pos, 'features': df_feat}}
  - df_pos columns:  ['track_id', 'frame_idx', 'pos_y', 'pos_x']
  - df_feat columns: ['frame_idx', 'confidence', 'bbox_x_min',
                      'bbox_x_max', 'bbox_y_min', 'bbox_y_max',
                      'bbox_area', 'bbox_aspect_ratio',
                      ...plus dynamic feature cols: area, eccentricity,
                      solidity, orientation, etc.]
  - For movement integration, we accept prediction_metadata.json as the
    file-based entry point (required by movement's ValidFile/GUI constraints),
    then derive the folder from Path(json_path).parent.
"""

import json
from pathlib import Path

import numpy as np
import xarray as xr
from octron import YOLO_results


def build_octron_dataset(file_path: Path, label_filter: str = "worm") -> xr.Dataset:

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Entry point not found: {file_path}\n"
            "Pass the path to prediction_metadata.json inside the OCTRON "
            "prediction subfolder."
        )
    if file_path.is_dir():
        raise ValueError(
            f"Expected a file (prediction_metadata.json), got a directory: {file_path}\n"
            "Pass the JSON file path, not the folder."
        )

    # YOLO_results expects the folder, not the JSON file.
    folder = file_path.parent
    yolo = YOLO_results(str(folder), verbose=False)

    # Filter track IDs to those matching label_filter.
    track_ids = [
        tid
        for tid, label in yolo.track_id_label.items()
        if label == label_filter
    ]
    if not track_ids:
        available = list(set(yolo.track_id_label.values()))
        raise FileNotFoundError(
            f"No tracks found for label='{label_filter}' in {folder}.\n"
            f"Available labels: {available}"
        )
    
    tracking = yolo.get_tracking_data(interpolate=False)

    matched = {tid: tracking[tid] for tid in track_ids if tid in tracking}
    if not matched:
        raise FileNotFoundError(
            f"label='{label_filter}' found in metadata but no tracking data "
            f"could be loaded from {folder}."
        )

    # Union of all frame indices across all matched tracks.
    all_frames = sorted(
        set().union(*[
            set(v["data"]["frame_idx"].astype(int))
            for v in matched.values()
        ])
    )
    n_frames = len(all_frames)
    n_ind    = len(matched)
    frame_map = {f: i for i, f in enumerate(all_frames)}

    # Pre-allocate with NaN — missing detections follow movement's convention.
    pos  = np.full((n_frames, 2, n_ind), np.nan, dtype="float32")
    shp  = np.full((n_frames, 2, n_ind), np.nan, dtype="float32")
    conf = np.full((n_frames, n_ind),    np.nan, dtype="float32")
    area = np.full((n_frames, n_ind),    np.nan, dtype="float32")
    ecc  = np.full((n_frames, n_ind),    np.nan, dtype="float32")
    sol  = np.full((n_frames, n_ind),    np.nan, dtype="float32")
    ori  = np.full((n_frames, n_ind),    np.nan, dtype="float32")

    individual_names = []

    for ii, (tid, track) in enumerate(sorted(matched.items())):
        # Name by label + track_id — unique even across multi-animal videos.
        individual_names.append(f"{track['label']}_track_{tid}")

        # df_pos confirmed columns from source: track_id, frame_idx, pos_y, pos_x
        df_pos = track["data"]
        frames = df_pos["frame_idx"].astype(int).values
        idx    = np.array([frame_map[f] for f in frames])

        pos[idx, 0, ii] = df_pos["pos_x"].values  # space[0] = x
        pos[idx, 1, ii] = df_pos["pos_y"].values  # space[1] = y
        df_feat     = track["features"]
        feat_frames = df_feat["frame_idx"].astype(int).values
        feat_idx    = np.array([frame_map[f] for f in feat_frames])

        if "bbox_x_min" in df_feat.columns and "bbox_x_max" in df_feat.columns:
            shp[feat_idx, 0, ii] = (
                df_feat["bbox_x_max"].values - df_feat["bbox_x_min"].values
            )
        if "bbox_y_min" in df_feat.columns and "bbox_y_max" in df_feat.columns:
            shp[feat_idx, 1, ii] = (
                df_feat["bbox_y_max"].values - df_feat["bbox_y_min"].values
            )

        # Optional columns — present in segmentation mode, absent in detection mode.
        if "confidence"   in df_feat.columns: conf[feat_idx, ii] = df_feat["confidence"].values
        if "area"         in df_feat.columns: area[feat_idx, ii] = df_feat["area"].values
        if "eccentricity" in df_feat.columns: ecc[feat_idx, ii]  = df_feat["eccentricity"].values
        if "solidity"     in df_feat.columns: sol[feat_idx, ii]  = df_feat["solidity"].values
        if "orientation"  in df_feat.columns: ori[feat_idx, ii]  = df_feat["orientation"].values

    # fps from prediction_metadata.json — users don't need to supply it manually.
    with open(file_path) as f:
        m = json.load(f)
    fps_val        = m.get("video_info", {}).get("fps_original")
    octron_version = m.get("octron_version", "unknown")

    time_coords = (
        np.array(all_frames) / fps_val
        if fps_val
        else np.array(all_frames, dtype="float64")
    )

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
            "space":       ["x", "y"],
            "individuals": individual_names,
        },
        attrs={
            "source_software": "OCTRON",
            "ds_type":         "bboxes",
            "fps":             fps_val,
            "label":           label_filter,
            "octron_version":  octron_version,
            "mask_store": str(folder / "predictions.zarr")
                if (folder / "predictions.zarr").exists()
                else None,
        },
    )