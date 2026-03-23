import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
import xarray as xr

from notes.utils import build_octron_dataset

def validate_loader_api():
    folder = Path("octron_predictions/worm_detailed_BotSort")
    ds = build_octron_dataset(folder / "prediction_metadata.json", "worm")
    
    # Known value from worm_track_1.csv frame 0:
    # pos_x=206.7516, pos_y=256.0156
    pos_0 = ds["position"].isel(time=0, individuals=0).values
    assert abs(pos_0[0] - 206.75) < 0.5, f"x wrong: {pos_0[0]}"
    assert abs(pos_0[1] - 256.01) < 0.5, f"y wrong: {pos_0[1]}"
    print(f"PASS: frame 0 centroid = ({pos_0[0]:.2f}, {pos_0[1]:.2f})")
    print("API Validation Passed: matches known OCTRON CSV value.")
    
if __name__ == "__main__":
    validate_loader_api()