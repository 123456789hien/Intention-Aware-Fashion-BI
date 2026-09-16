from __future__ import annotations
from pathlib import Path
import pandas as pd
from .data_loader import ROOT

REQUIRED = ["three_tower_model.pt", "visual_features.npy", "semantic_features.npy"]


def artifact_status() -> dict[str, bool]:
    return {name: (ROOT / "models" / name).exists() or (ROOT / "data" / name).exists() for name in REQUIRED}


def can_infer() -> bool:
    # The supplied Drive package has no lda_article_index.csv. The arrays are full-population
    # features, while the app catalog is a sample. Without a verified article-to-row mapping,
    # producing a score would be unsafe even when the three binary artifacts exist.
    return False


def explain_unavailable() -> str:
    missing = [n for n, ok in artifact_status().items() if not ok]
    if missing:
        return "Actual Three-Tower inference is locked because these thesis artifacts are missing: " + ", ".join(missing)
    return "Actual Three-Tower inference is locked because the supplied package has no verified article-to-feature row mapping. No lda_article_index.csv is required or assumed."


def rank_with_actual_model(*args, **kwargs) -> pd.DataFrame:
    """Reserved for the exact thesis checkpoint and preprocessing contract.

    It intentionally fails closed. A score must never be fabricated from catalog metadata.
    Implement only after the real checkpoint, feature arrays, index and preprocessing code
    are supplied and verified against the training notebook.
    """
    if not can_infer(): raise RuntimeError(explain_unavailable())
    raise NotImplementedError("Connect the verified thesis checkpoint and preprocessing contract here.")
