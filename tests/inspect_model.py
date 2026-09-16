from pathlib import Path
import sys, json
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.data_loader import DATA_DIR

p=DATA_DIR/'three_tower_model.pt'
obj=torch.load(p, map_location='cpu', weights_only=False)
print('checkpoint_type:', type(obj).__name__)
if isinstance(obj, dict):
    print('top_keys:', list(obj.keys()))
    for key,value in obj.items():
        if isinstance(value, dict):
            print(key, 'keys:', list(value.keys())[:30])
            if key in ('state_dict','model_state_dict'):
                for n,t in list(value.items())[:40]: print(' ',n, tuple(t.shape) if hasattr(t,'shape') else type(t).__name__)
        elif hasattr(value,'shape'):
            print(key, 'shape:', tuple(value.shape))
labels=json.loads((DATA_DIR/'intention_labels.json').read_text())
print('labels_type:', type(labels).__name__)
print('labels:', json.dumps(labels, ensure_ascii=False)[:1200])
