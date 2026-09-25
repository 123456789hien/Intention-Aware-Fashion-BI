"""
models.py
================================================================================
Matches EXACTLY the ThreeTowerModel / TwoTowerModel architecture defined in
section-12-three-tower-neural-network-training.ipynb and
section-13-two-tower-baseline-training.ipynb of the thesis. Do not change any
dimension — doing so will make the .pt checkpoints fail to load (state_dict
shape mismatch).

  Tower 1 (Visual):    2048 -> 512 -> 256
  Tower 2 (Semantic):  387  -> 512 -> 256   (384 SBERT + 3 demo)
  Tower 3 (Intention): 10   -> 64  -> 32    (Three-Tower only)
  Fusion (3T):         544  -> 256 -> 128 -> 1  (256+256+32)
  Fusion (2T):         512  -> 256 -> 128 -> 1  (256+256)
================================================================================
"""

import torch
import torch.nn as nn
import streamlit as st

DIM_VISUAL = 2048
DIM_SEMANTIC = 384
DIM_USER_DEMO = 3
DIM_INTENTION = 10


class ThreeTowerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tower1 = nn.Sequential(
            nn.Linear(DIM_VISUAL, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU()
        )
        self.tower2 = nn.Sequential(
            nn.Linear(DIM_SEMANTIC + DIM_USER_DEMO, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU()
        )
        self.tower3 = nn.Sequential(
            nn.Linear(DIM_INTENTION, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU()
        )
        self.predictor = nn.Sequential(
            nn.Linear(256 + 256 + 32, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid()
        )

    def forward(self, visual, semantic, demo, item_int, user_int):
        t1 = self.tower1(visual)
        t2 = self.tower2(torch.cat([semantic, demo], dim=1))
        t3 = self.tower3(item_int * user_int)  # Hadamard alignment
        fused = torch.cat([t1, t2, t3], dim=1)
        return self.predictor(fused).squeeze(1), (item_int * user_int)


class TwoTowerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tower1 = nn.Sequential(
            nn.Linear(DIM_VISUAL, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU()
        )
        self.tower2 = nn.Sequential(
            nn.Linear(DIM_SEMANTIC + DIM_USER_DEMO, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU()
        )
        self.predictor = nn.Sequential(
            nn.Linear(256 + 256, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid()
        )

    def forward(self, visual, semantic, demo, item_int, user_int):
        t1 = self.tower1(visual)
        t2 = self.tower2(torch.cat([semantic, demo], dim=1))
        return self.predictor(torch.cat([t1, t2], dim=1)).squeeze(1)


@st.cache_resource(show_spinner="🧠 Loading Three-Tower & Two-Tower models...")
def load_models(three_tower_path: str, two_tower_path: str):
    three_model = ThreeTowerModel()
    two_model = TwoTowerModel()

    three_model.load_state_dict(torch.load(three_tower_path, map_location="cpu"))
    two_model.load_state_dict(torch.load(two_tower_path, map_location="cpu"))

    three_model.eval()
    two_model.eval()
    return three_model, two_model
