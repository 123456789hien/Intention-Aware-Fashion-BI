# Intention-Aware Fashion Recommender — Prototype Demo

An end-to-end prototype accompanying the Master's thesis **"Deep
Learning-Driven Business Intelligence for Personalized Fashion Retail:
Integrating Intention Analytics and Recommendation System"** (Do Thi Hien,
2026) — bringing the **Three-Tower Neural Network** architecture to life as
an interactive web app, built with [Streamlit](https://streamlit.io).

**➡️ Live demo:** `https://<your-app-name>.streamlit.app` (fill in after deploying)

---

## The business value this project addresses

Traditional fashion e-commerce recommenders answer *"what does the customer
buy"* and *"who is the customer"* well, but miss *"why does the customer
buy"* — the real motivation behind the purchase decision. This app visually
demonstrates how a third tower, accounting for just **0.2% of total model
parameters**, produces a measurable **+3.52% AUC** improvement — especially
strong in segments with narrow, well-defined shopping intentions (baby care,
personal intimate wear, professional menswear) — while also generating
Business Intelligence outputs (intention-based segmentation, supply-demand
gap reporting, personalisation budget prioritisation) that feed directly
into buying and marketing decisions.

## App structure

| Page | Content | Corresponding thesis chapter |
|---|---|---|
| 🏠 Home | Overview, business value | Abstract, Chapter 1 |
| 🛍️ Product Catalog | Browse products by 10 shopping-intention groups, with real images | Chapter 4.2 (Intention Discovery) |
| 🎯 Recommendation Demo | Compare Three-Tower vs Two-Tower, with Hadamard-alignment explanations | Chapter 4.5 (Comparative Evaluation) |
| 📊 Business Intelligence | Segmentation, supply-demand gap, investment priority | Chapter 5 |
| ℹ️ Methodology | Model architecture, data source, sampling method | Chapter 3, Appendix |

## Data architecture — why code (GitHub) and data (Google Drive) are separated

```
GitHub repo (code, <5MB)          Google Drive (1 file: data_export.zip, ~120-150MB)
├── app.py                        ├── articles_sample.csv
├── pages/                        ├── visual_features_sample.npy
├── utils/                        ├── semantic_features_sample.npy
├── requirements.txt              ├── demo_personas.csv
└── .streamlit/                   ├── three_tower_best.pt / two_tower_best.pt
                                   └── images/*.jpg  (~5,300 images, resized to 400px)
        │                                    │
        └──────── app downloads & extracts on startup ──┘
                   (utils/data_loader.py: gdown.download + zipfile, cached once)
```

Product images are resized to a maximum of 400px on the longest edge
(Script 01, Step 6b) — a ~96% size reduction (1.5GB → 55MB) with no loss of
usable display quality, since ResNet-50 (Chapter 3) itself only uses
224×224px images when extracting visual features.

Data is a **stratified 5% sample per dominant intention** — the exact same
method and rate validated in thesis Section 11 (distribution deviation
MAD ≤ 0.0036pp from the true population). See
[`01_sample_and_export.py`](../01_sample_and_export.py) and
[`02_DRIVE_UPLOAD_GUIDE.md`](../02_DRIVE_UPLOAD_GUIDE.md) at the project
root for how this sample was created.

## Running locally

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt

mkdir -p .streamlit
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Open .streamlit/secrets.toml and paste your real GDRIVE_FILE_ID

streamlit run app.py
```

The first run will automatically download the dataset from Google Drive
into a `data/` folder (1–3 minutes depending on connection speed); later
runs reuse the cache.

## Deploy to Streamlit Community Cloud

1. Push the code (without `data/`, already excluded by `.gitignore`) to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select your repo, branch `main`, main file path: `app.py`.
4. Before deploying, open **Advanced settings → Secrets** and paste:
   ```toml
   GDRIVE_FILE_ID = "1AbCdEfGhIjKlMnOpQrStUvWxYz"
   ```
   (The File ID of `data_export.zip` — a single zip file, not a folder ID —
   see `02_DRIVE_UPLOAD_GUIDE.md` at the project root.)
5. Click **Deploy**. Watch the logs — the first cold start will take a few
   minutes while `gdown` downloads the dataset from Drive; subsequent visits
   (same live container) will be fast thanks to `st.cache_resource`.

## System requirements

- Python 3.10+
- See `requirements.txt` (PyTorch CPU-only is sufficient for inference — no GPU needed)

## Directory structure

```
app/
├── app.py                        # Home page
├── requirements.txt
├── .gitignore
├── .streamlit/
│   ├── config.toml               # Theme
│   └── secrets.toml.example
├── utils/
│   ├── data_loader.py            # Downloads & caches data from Google Drive
│   ├── models.py                 # Three-Tower / Two-Tower architecture
│   └── recommender.py            # Recommendation logic + explanations
└── pages/
    ├── 1_Product_Catalog.py
    ├── 2_Recommendation_Demo.py
    ├── 3_Business_Intelligence.py
    └── 4_About_Methodology.py
```

## Citation

If you reference this prototype, please cite the original thesis:

> Do, T. H. (2026). *Deep Learning-Driven Business Intelligence for
> Personalized Fashion Retail: Integrating Intention Analytics and
> Recommendation System* [Master's thesis, National Research University
> Higher School of Economics].

Dataset: H&M Group (2022). *H&M Personalized Fashion Recommendations*
[Data set]. Kaggle.
