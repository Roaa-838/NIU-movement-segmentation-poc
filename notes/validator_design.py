 # Test 4: Confirms ValidFile rejects directories — loader must accept metadata JSON, not folder

import tempfile, os
from pathlib import Path
from movement.validators.files import ValidFile

# Create a temp directory
with tempfile.TemporaryDirectory() as tmpdir:
    try:
        vf = ValidFile(tmpdir)
        print("FAIL: ValidFile accepted a directory — our validator design is wrong")
    except Exception as e:
        print(f"PASS: ValidFile rejects directories — {type(e).__name__}: {e}")
        print("Confirmed: loader must accept metadata JSON, not folder path")