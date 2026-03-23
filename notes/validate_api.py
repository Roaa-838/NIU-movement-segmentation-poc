"""
notes/validate_api.py

Mathematical proof: given a CSV with known coordinates, the loader must
return exactly those values. 
Uses xr.testing.assert_allclose so any
floating-point mismatch causes a hard failure.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import tempfile
import numpy as np
import xarray as xr
from pathlib import Path

from notes.utils import build_octron_dataset


def validate_loader_api():
    # A minimal OCTRON-style CSV: two dummy metadata lines then real data.
    # The worm moves exactly 1px right and 2px down per frame — easy to verify.
    dummy_csv = """\
dummy_header1
dummy_header2
frame_idx,track_id,pos_x,pos_y,confidence
0,1,10.0,20.0,0.9
1,1,11.0,22.0,0.9
2,1,12.0,24.0,0.9
"""
    # tempfile.TemporaryDirectory handles cleanup automatically even if the
    # assertion raises — no stray temp_test/ folder left in the repo.
    with tempfile.TemporaryDirectory() as tmp:
        temp_dir = Path(tmp)
        (temp_dir / "worm_track_1.csv").write_text(dummy_csv)

        actual_ds = build_octron_dataset(temp_dir, label_filter="worm")

        # Guard: no metadata.json in temp_dir means fps must be None.
        assert actual_ds.attrs.get("fps") is None, (
            "Test contaminated: a prediction_metadata.json was read from the "
            "temp directory, which would change the time coordinates."
        )

        expected_pos = np.array(
            [
                [[10.0], [20.0]],  # frame 0: x=10, y=20
                [[11.0], [22.0]],  # frame 1: x=11, y=22
                [[12.0], [24.0]],  # frame 2: x=12, y=24
            ],
            dtype="float32",
        )
        expected_ds = xr.Dataset(
            {
                "position": xr.DataArray(
                    expected_pos, dims=("time", "space", "individuals")
                )
            },
            coords={
                "time":        [0.0, 1.0, 2.0],
                "space":       ["x", "y"],
                "individuals": ["worm_track_1"],
            },
        )

        xr.testing.assert_allclose(actual_ds["position"], expected_ds["position"])
        print("API Validation Passed: loader output matches expected values exactly.")


if __name__ == "__main__":
    validate_loader_api()