import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from pathlib import Path
from movement.kinematics import compute_velocity
from notes.utils import build_octron_dataset


def validate_kinematics(ds):
    vel = compute_velocity(ds["position"])
    
    # Calculate speed (magnitude of the velocity vector)
    # axis 1 is the space dimension (x, y)
    speed = np.linalg.norm(vel.values, axis=1) 
    
    max_speed = np.nanmax(speed)
    mean_speed = np.nanmean(speed)
    
    print("--- Kinematic Sanity Check ---")
    print(f"Max Speed:  {max_speed:.2f} px/s")
    print(f"Mean Speed: {mean_speed:.2f} px/s")
    
    # Basic threshold check to ensure we aren't getting garbage values
    assert max_speed < 10000, "Velocity out of bounds. Check FPS or unit conversions."
    print("Velocity is within realistic bounds.")

if __name__ == "__main__":
    folder_path = Path("octron_predictions/worm_detailed_BotSort")
    
    print("Loading dataset...")
    dataset = build_octron_dataset(folder_path, "worm")
    
    validate_kinematics(dataset)