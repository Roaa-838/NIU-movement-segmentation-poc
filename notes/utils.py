"""
notes/utils.py

Shared loader utilities used by all validation and test scripts.

Two public functions:
  read_octron_csv()      — reads a single OCTRON track CSV, skipping the
                           variable-length metadata header.
  build_octron_dataset() — builds a movement-compatible xr.Dataset from
                           all per-class CSVs in an OCTRON prediction folder.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


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
        print(f"  Skipped {header_idx} metadata lines in {filepath.name}")

    return pd.read_csv(filepath, skiprows=header_idx)


def build_octron_dataset(folder: Path, label_filter: str = "worm") -> xr.Dataset:

    pattern = f"{label_filter}_track_*.csv"
    csv_files = sorted(folder.glob(pattern))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files for label='{label_filter}' found in {folder}"
        )

    dfs = [read_octron_csv(f) for f in csv_files]

    # Detect which column holds the frame number — OCTRON has used both names.
    frame_col = next(
        (c for c in ["frame_idx", "frame_counter", "frame"] if c in dfs[0].columns),
        None,
    )
    if not frame_col:
        raise ValueError(
            "Could not find a valid frame column (e.g. 'frame_idx') in the CSV."
        )

    # Union of all frame numbers across individuals, sorted.
    all_frames = sorted(set().union(*[set(df[frame_col].astype(int)) for df in dfs]))
    n_frames = len(all_frames)
    n_ind = len(dfs)
    frame_map = {f: i for i, f in enumerate(all_frames)}

    # Pre-allocate with NaN so missing detections follow movement's convention.
    pos  = np.full((n_frames, 2, n_ind), np.nan, "float32")
    shp  = np.full((n_frames, 2, n_ind), np.nan, "float32")
    conf = np.full((n_frames, n_ind),    np.nan, "float32")
    area = np.full((n_frames, n_ind),    np.nan, "float32")
    ecc  = np.full((n_frames, n_ind),    np.nan, "float32")
    sol  = np.full((n_frames, n_ind),    np.nan, "float32")
    ori  = np.full((n_frames, n_ind),    np.nan, "float32")

    has_minmax = "bbox_x_min" in dfs[0].columns

    # Vectorized array population (replaces .iterrows — ~100x faster at scale).
    for ii, df in enumerate(dfs):
        row_frames = df[frame_col].astype(int).values
        idx = np.array([frame_map[f] for f in row_frames])

        pos[idx, 0, ii] = df["pos_x"].values
        pos[idx, 1, ii] = df["pos_y"].values

        if has_minmax:
            shp[idx, 0, ii] = (df["bbox_x_max"] - df["bbox_x_min"]).values
            shp[idx, 1, ii] = (df["bbox_y_max"] - df["bbox_y_min"]).values

        # Safe column checks — not all OCTRON modes export these.
        if "confidence"   in df.columns: conf[idx, ii] = df["confidence"].values
        if "area"         in df.columns: area[idx, ii] = df["area"].values
        if "eccentricity" in df.columns: ecc[idx, ii]  = df["eccentricity"].values
        if "solidity"     in df.columns: sol[idx, ii]  = df["solidity"].values
        if "orientation"  in df.columns: ori[idx, ii]  = df["orientation"].values

    # Read fps from metadata — users don't need to supply it manually.
    meta_path = folder / "prediction_metadata.json"
    fps_val = None
    octron_version = "unknown"
    if meta_path.exists():
        with open(meta_path) as f:
            m = json.load(f)
        fps_val = m.get("video_info", {}).get("fps_original")
        octron_version = m.get("octron_version", "unknown")

    # Time in seconds if fps is known, else raw frame numbers.
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
            "individuals": [f.stem for f in csv_files],
        },
        attrs={
            "source_software": "OCTRON",
            "fps":             fps_val,
            "label":           label_filter,
            "octron_version":  octron_version,
            "mask_store": str(folder / "predictions.zarr")
                if (folder / "predictions.zarr").exists()
                else None,
        },
    )