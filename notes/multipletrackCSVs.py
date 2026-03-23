# notes/multipletrackCSVs.py
#
# NOTE: This was the initial approach that failed on OCTRON's header lines.
# A plain pd.read_csv() call produces a ParserError because OCTRON CSVs have
# 7 metadata lines before the real column headers.
# The fixed version is read_octron_csv() in notes/utils.py.
#
# This file is kept as an honest record of what I discovered during testing.

import glob
import numpy as np
import xarray as xr
from pathlib import Path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from notes.utils import read_octron_csv, build_octron_dataset

octron_folder = Path("octron_predictions/worm_detailed_BotSort")

# Use the header-aware reader — plain pd.read_csv(f) fails here.
csv_files = sorted(glob.glob(str(octron_folder / "*.csv")))
dfs = [read_octron_csv(Path(f)) for f in csv_files]

print(f"Loaded {len(dfs)} CSVs")
print(f"Columns: {list(dfs[0].columns)}")

# The cleaner approach: use build_octron_dataset directly.
ds = build_octron_dataset(octron_folder, label_filter="worm")
print(ds)

from movement.kinematics import compute_velocity
vel = compute_velocity(ds["position"])
print(f"\ncompute_velocity works: {vel.shape}")