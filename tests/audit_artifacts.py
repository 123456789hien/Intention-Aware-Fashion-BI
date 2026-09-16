from pathlib import Path
import sys
import json
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_loader import sync_from_drive, DATA_DIR

status = sync_from_drive(timeout=180)
print(json.dumps(status, indent=2))
for name in ["visual_features.npy", "semantic_features.npy", "three_tower_model.pt"]:
    path = DATA_DIR / name
    print(f"{name}: exists={path.exists()} bytes={path.stat().st_size if path.exists() else 0}")
    if path.exists() and name.endswith(".npy"):
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        print(f"  shape={arr.shape} dtype={arr.dtype}")
