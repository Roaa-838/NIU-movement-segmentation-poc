import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import shutil
import tempfile
from pathlib import Path

from notes.utils import build_octron_dataset 


def validate_edge_cases(folder_path: Path):
    print("--- Testing Edge Cases ---")
    
    try:
        bad_ds = build_octron_dataset(folder_path, label_filter="dragon")
        print("FAIL: Script should have crashed on a missing label but didn't.")
    except FileNotFoundError as e:
        print(f"PASS: Correctly caught missing label. ({e})")
        
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_folder = Path(temp_dir)
        
        src_csv = folder_path / "worm_track_1.csv"
        dst_csv = temp_folder / "worm_track_1.csv"
        
        if src_csv.exists():
            shutil.copy(src_csv, dst_csv)
            ds_no_meta = build_octron_dataset(temp_folder, label_filter="worm")
            
            if ds_no_meta.attrs.get("fps") is None:
                print("PASS: Dataset gracefully handled missing prediction_metadata.json.")
            else:
                print("FAIL: Dataset did not handle missing metadata correctly.")
        else:
            # We print SKIP instead of FAIL because the absence of the sample CSV 
            # is a local environment issue, not a failure of the loader logic.
            print(f"SKIP: Could not find {src_csv.name} in {folder_path} to run Test 2.")

if __name__ == "__main__":
    base_folder = Path("octron_predictions/worm_detailed_BotSort")
    validate_edge_cases(base_folder)