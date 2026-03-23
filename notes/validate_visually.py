import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent)) 
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

from notes.utils import build_octron_dataset

def validate_visually(video_path: Path, ds, frame_idx: int = 0):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    
    if not ret:
        print(f"Failed to read frame {frame_idx} from {video_path}")
        return
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    fps = ds.attrs.get("fps") or 1.0 
    time_sec = frame_idx / fps
    
    pos = ds["position"].sel(time=time_sec, method="nearest").isel(individuals=0).values
    
    plt.figure(figsize=(10, 10))
    plt.imshow(frame_rgb)
    plt.scatter(pos[0], pos[1], c='red', s=100, label='OCTRON Centroid')
    
    plt.title(f"Visual Validation: Frame {frame_idx}")
    plt.legend()
    plt.axis('off')
    
    out_dir = Path("docs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"visual_validation_frame{frame_idx}.png"
    
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    print(f"Saved validation plot to: {out_path}")
    
    plt.show()

if __name__ == "__main__":
    BASE_DIR = Path("octron_predictions")
    FOLDER = BASE_DIR / "worm_detailed_BotSort"
    VIDEO_PATH = BASE_DIR / "worm_detailed.mp4"
    
    print("Building dataset...")
    dataset = build_octron_dataset(FOLDER, "worm")
    
    print("Running visual validation...")
    validate_visually(VIDEO_PATH, dataset, frame_idx=0)