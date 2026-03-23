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
        bad_ds = build_octron_dataset(folder_path / "prediction_metadata.json", label_filter="dragon")
        print("FAIL: Script should have crashed on a missing label but didn't.")
    except FileNotFoundError as e:
        print(f"PASS: Correctly caught missing label. ({e})")
        
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_folder = Path(temp_dir)
        src_csv = folder_path / "worm_track_1.csv"
        dst_csv = temp_folder / "worm_track_1.csv"
        
        if src_csv.exists():
            import shutil
            shutil.copy(src_csv, dst_csv)
            # Test 2: If we pass a JSON path that doesn't exist, it should fail immediately
            fake_json_path = temp_folder / "prediction_metadata.json"
            try:
                ds_no_meta = build_octron_dataset(fake_json_path, label_filter="worm")
                print("FAIL: Dataset loaded even though entry point JSON was missing.")
            except FileNotFoundError:
                print("PASS: Dataset correctly rejected missing prediction_metadata.json entry point.")
        else:
            print(f"SKIP: Could not find {src_csv.name} in {folder_path} to run Test 2.")
            
if __name__ == "__main__":
    base_folder = Path("octron_predictions/worm_detailed_BotSort")
    validate_edge_cases(base_folder)