"""
data_loader.py
================================================================================
Downloads the full demo dataset (5%/intention sample from Script 01) from
Google Drive to the Streamlit container's local disk, once per cold
container start, using st.cache_resource.

Why st.cache_resource instead of st.cache_data:
- cache_resource keeps the object (including the fact that files were
  already downloaded to disk) alive for the container's lifetime, without
  re-serializing/re-hashing on every call — a better fit for "download once"
  than for "cache a computed value".
================================================================================
"""

import os
import json
import zipfile
import streamlit as st
import pandas as pd
import numpy as np
import gdown

DATA_DIR = "data"
ZIP_TMP_PATH = "data_export.zip"


@st.cache_resource(show_spinner="📥 Downloading demo data from Google Drive (one-time only)...")
def download_data():
    """
    Downloads a SINGLE .zip file (data_export.zip, ~120-150MB after image
    resizing) from Google Drive and extracts it into `data/`. Using one zip
    file instead of gdown.download_folder (which downloads ~5,300 individual
    files) is far more stable and faster — download_folder is prone to
    Google's rate limiting/hanging with large file counts.

    Idempotent: if `data/` already has all the core files within the same
    session/container, skips re-downloading.
    """
    core_files = [
        "articles_sample.csv",
        "article_intention_profiles_sample.csv",
        "article_index_sample.csv",
        "visual_features_sample.npy",
        "semantic_features_sample.npy",
        "demo_personas.csv",
        "intention_labels.json",
        "sampling_report.json",
        "three_tower_best.pt",
        "two_tower_best.pt",
    ]

    already_ok = os.path.isdir(DATA_DIR) and all(
        os.path.exists(os.path.join(DATA_DIR, f)) for f in core_files
    )
    if already_ok:
        return True

    file_id = st.secrets.get("GDRIVE_FILE_ID", None)
    if not file_id or file_id.startswith("PASTE_"):
        st.error(
            "⚠️ `GDRIVE_FILE_ID` is not configured in Streamlit Secrets. "
            "See README.md, section 'Deploy to Streamlit', for setup instructions."
        )
        st.stop()

    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(ZIP_TMP_PATH):
        gdown.download(id=file_id, output=ZIP_TMP_PATH, quiet=False)

    if not os.path.exists(ZIP_TMP_PATH) or os.path.getsize(ZIP_TMP_PATH) < 1_000_000:
        st.error(
            "❌ Zip download failed or the file is too small (Google Drive may "
            "have returned a virus-scan warning page instead of the real file — "
            "this sometimes happens with large public files missing a confirm "
            "token). Double-check GDRIVE_FILE_ID and the sharing permission "
            "(Anyone with the link – Viewer)."
        )
        st.stop()

    with zipfile.ZipFile(ZIP_TMP_PATH, "r") as zf:
        zf.extractall(DATA_DIR)

    os.remove(ZIP_TMP_PATH)  # free up disk space right after extraction

    missing = [f for f in core_files if not os.path.exists(os.path.join(DATA_DIR, f))]
    if missing:
        st.error(
            f"❌ Extraction incomplete, missing: {missing}. "
            "Check the zip file structure (these files must sit at the zip "
            "root, not nested inside an extra subfolder)."
        )
        st.stop()

    return True


@st.cache_data(show_spinner=False)
def load_articles() -> pd.DataFrame:
    download_data()
    df = pd.read_csv(os.path.join(DATA_DIR, "articles_sample.csv"))
    df["article_id"] = df["article_id"].astype(str).str.zfill(10)
    return df


@st.cache_data(show_spinner=False)
def load_article_intention_profiles() -> pd.DataFrame:
    download_data()
    df = pd.read_csv(os.path.join(DATA_DIR, "article_intention_profiles_sample.csv"))
    df["article_id"] = df["article_id"].astype(str).str.zfill(10)
    return df


@st.cache_resource(show_spinner=False)
def load_feature_matrices():
    download_data()
    visual = np.load(os.path.join(DATA_DIR, "visual_features_sample.npy")).astype(np.float32)
    semantic = np.load(os.path.join(DATA_DIR, "semantic_features_sample.npy")).astype(np.float32)
    idx_map = pd.read_csv(os.path.join(DATA_DIR, "article_index_sample.csv"))
    idx_map["article_id"] = idx_map["article_id"].astype(str).str.zfill(10)
    art_feat_idx = dict(zip(idx_map["article_id"], idx_map["feature_index"]))
    return visual, semantic, art_feat_idx


@st.cache_data(show_spinner=False)
def load_demo_personas() -> pd.DataFrame:
    download_data()
    df = pd.read_csv(os.path.join(DATA_DIR, "demo_personas.csv"))
    df["customer_id"] = df["customer_id"].astype(str)
    return df


@st.cache_data(show_spinner=False)
def load_intention_labels() -> dict:
    download_data()
    with open(os.path.join(DATA_DIR, "intention_labels.json")) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_sampling_report() -> dict:
    download_data()
    with open(os.path.join(DATA_DIR, "sampling_report.json")) as f:
        return json.load(f)


def image_path(article_id: str) -> str:
    """Local image path for a given article_id, once downloaded from Drive."""
    return os.path.join(DATA_DIR, "images", f"{article_id}.jpg")


def model_paths():
    download_data()
    return (
        os.path.join(DATA_DIR, "three_tower_best.pt"),
        os.path.join(DATA_DIR, "two_tower_best.pt"),
    )
