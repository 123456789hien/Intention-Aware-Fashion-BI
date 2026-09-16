from __future__ import annotations
from pathlib import Path
import pandas as pd
from .data_loader import ROOT

REQUIRED = ["three_tower_model.pt", "visual_features.npy", "semantic_features.npy", "lda_article_index.csv"]


def artifact_status() -> dict[str, bool]:
    return {name: (ROOT / "models" / name).exists() or (ROOT / "data" / name).exists() for name in REQUIRED}


def can_infer() -> bool:
    return all(artifact_status().values())


def explain_unavailable() -> str:
    missing = [n for n, ok in artifact_status().items() if not ok]
    return "Actual Three-Tower inference is locked because these thesis artifacts are missing: " + ", ".join(missing)


def rank_with_actual_model(*args, **kwargs) -> pd.DataFrame:
    """Reserved for the exact thesis checkpoint and preprocessing contract.

    It intentionally fails closed. A score must never be fabricated from catalog metadata.
    Implement only after the real checkpoint, feature arrays, index and preprocessing code
    are supplied and verified against the training notebook.
    """
    if not can_infer(): raise RuntimeError(explain_unavailable())
    raise NotImplementedError("Connect the verified thesis checkpoint and preprocessing contract here.")
