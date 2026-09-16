from __future__ import annotations
import json, hashlib, time, re
from functools import lru_cache
from pathlib import Path
from typing import Any
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "data_sources.json"
DATA_DIR = ROOT / "data"


def _config() -> dict[str, Any]:
    text = CONFIG.read_text(encoding="utf-8")
    text = text.split("\n\n//", 1)[0]
    return json.loads(text)


def drive_download_url(file_id: str) -> str:
    return f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"


def sync_from_drive(timeout: int = 90, force: bool = False) -> dict[str, str]:
    """Download only real configured Drive files. Never creates fallback data."""
    DATA_DIR.mkdir(exist_ok=True)
    status: dict[str, str] = {}
    for key, item in _config()["sources"].items():
        file_id, filename = item.get("file_id", ""), item["filename"]
        dest = DATA_DIR / filename
        if not file_id:
            status[key] = "missing_file_id"
            continue
        if key in {"three_tower_model", "visual_features", "semantic_features"} and not force:
            status[key] = "deferred_until_artifact_request"
            continue
        if dest.exists() and not force:
            status[key] = f"cached:{dest.stat().st_size}"
            continue
        for attempt in range(3):
            try:
                r = requests.get(drive_download_url(file_id), timeout=timeout)
                r.raise_for_status()
                if b"<html" in r.content[:500].lower() and "404" in r.text[:1000]:
                    status[key] = "drive_404"
                    break
                dest.write_bytes(r.content)
                status[key] = f"downloaded:{len(r.content)}"
                break
            except requests.RequestException as exc:
                status[key] = f"error:{exc}"
                if attempt < 2: time.sleep(2 ** attempt)
    return status


def load_json(name: str) -> dict[str, Any]:
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


def load_csv(name: str, **kwargs) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    return pd.read_csv(path, **kwargs)


def load_bundle() -> dict[str, Any]:
    return {
        "summary": load_json("app_summary.json"),
        "catalog": load_csv("article_metadata.csv", dtype={"article_id": str}),
        "article_intentions": load_csv("article_intention_profiles.csv", dtype={"article_id": str}),
        "customers": load_csv("customers_cleaned.csv", dtype={"customer_id": str}),
        "images": load_csv("image_mapping.csv", dtype={"article_id": str}),
        "sampling": load_csv("sampling_stats.csv"),
        "interactions": load_csv("test_interactions.csv", dtype={"customer_id": str, "article_id": str}),
        "confidence": load_csv("user_confidence_scores.csv", dtype={"customer_id": str}),
        "user_dist": load_csv("user_dominant_intention_dist.csv"),
        "user_weights": load_csv("user_intention_weights.csv", dtype={"customer_id": str}),
    }


def validate_bundle(bundle: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    catalog, prof, images = bundle["catalog"], bundle["article_intentions"], bundle["images"]
    if catalog["article_id"].duplicated().any(): errors.append("article_metadata has duplicate article_id")
    if prof["article_id"].duplicated().any(): errors.append("article_intention_profiles has duplicate article_id")
    if not set(prof.article_id).issubset(set(catalog.article_id)): errors.append("intention profile contains unknown article_id")
    if not set(images.article_id).issubset(set(catalog.article_id)): errors.append("image mapping contains unknown article_id")
    intention_cols = [c for c in prof.columns if c.startswith("intention_") and c[10:].isdigit()]
    if not intention_cols: errors.append("no intention weight columns found")
    else:
        vals = prof[intention_cols].apply(pd.to_numeric, errors="coerce")
        if vals.isna().any().any() or ((vals < 0) | (vals > 1)).any().any(): errors.append("invalid article intention probabilities")
    return errors


def local_image_path(image_id: str) -> Path | None:
    candidates = [ROOT / "assets" / "product_images" / f"{image_id}.jpg", DATA_DIR / "images" / f"{image_id}.jpg"]
    return next((p for p in candidates if p.exists()), None)


@lru_cache(maxsize=1)
def drive_image_map() -> dict[str, str]:
    folder_url = _config()["images"]["folder_url"]
    folder_id = folder_url.split("/folders/")[1].split("?")[0]
    html = requests.get(f"https://drive.google.com/drive/folders/{folder_id}?usp=drive_link", timeout=60).text
    pattern = re.compile(r'aria-label="([0-9]{10}\.jpg) Image Shared".*?data-id="([A-Za-z0-9_-]+)"', re.S)
    return dict(pattern.findall(html))


def image_url(image_id: str) -> str | None:
    path = local_image_path(image_id)
    if path:
        return str(path)
    file_id = drive_image_map().get(f"{image_id}.jpg")
    return f"https://drive.google.com/thumbnail?id={file_id}&sz=w800" if file_id else None
