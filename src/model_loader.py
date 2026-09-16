from __future__ import annotations
from pathlib import Path
import torch
from torch import nn
from .data_loader import ROOT

class ThreeTowerNetwork(nn.Module):
    """Architecture reconstructed from the supplied checkpoint state_dict."""
    def __init__(self, dim_visual=2048, dim_semantic=384, dim_demo=3, dim_intention=10):
        super().__init__()
        self.tower1 = nn.Sequential(nn.Linear(dim_visual, 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, 256), nn.ReLU())
        self.tower2 = nn.Sequential(nn.Linear(dim_semantic + dim_demo, 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, 256), nn.ReLU())
        self.tower3 = nn.Sequential(nn.Linear(dim_intention, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 32), nn.ReLU())
        self.predictor = nn.Sequential(nn.Linear(544, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, visual, semantic, demographic, item_intention, user_intention):
        v = self.tower1(visual)
        s = self.tower2(torch.cat([semantic, demographic], dim=1))
        i = self.tower3(item_intention * user_intention)
        return self.predictor(torch.cat([v, s, i], dim=1)).squeeze(-1)


def checkpoint_path() -> Path:
    for p in [ROOT / "models" / "three_tower_model.pt", ROOT / "data" / "three_tower_model.pt"]:
        if p.exists(): return p
    raise FileNotFoundError("three_tower_model.pt is not available")


def load_three_tower() -> tuple[ThreeTowerNetwork, dict]:
    checkpoint = torch.load(checkpoint_path(), map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    model = ThreeTowerNetwork(
        int(config.get("dim_visual", 2048)), int(config.get("dim_semantic", 384)),
        int(config.get("dim_demo", 3)), int(config.get("dim_intention", 10))
    )
    state = checkpoint.get("model_state") or checkpoint.get("state_dict") or checkpoint.get("model_state_dict")
    if state is None: raise ValueError("Checkpoint has no model_state/state_dict")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, checkpoint
