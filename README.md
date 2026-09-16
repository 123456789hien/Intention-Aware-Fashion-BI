# Intention-Aware Fashion BI

A Streamlit business-intelligence application for the H&M fashion thesis. The application connects purchase-intention profiling, product intelligence, customer segmentation, catalogue representation and Three-Tower model evidence.

> The system moves fashion recommendation from understanding **what** customers buy to understanding **why** they buy.

## Data integrity policy

This repository does not contain demo rows, random values or simulated recommendation scores. At runtime, `src/data_loader.py` downloads the supplied CSV/JSON files from Google Drive using the file IDs in `config/data_sources.json`. The app validates keys and probability columns before rendering.

The recommendation page fails closed until the exact trained artifacts are supplied:

- `three_tower_model.pt`
- `visual_features.npy`
- `semantic_features.npy`
- `lda_article_index.csv`

Those artifacts are intentionally not invented. They must be added to `config/data_sources.json` with their real Google Drive file IDs, or placed in the configured runtime data/model directories. The inference preprocessing must also be verified against the training notebook before enabling scores.

The supplied metadata does not currently include product prices. The application therefore displays `Price unavailable in supplied Google Drive data`; it never fabricates a price. Add a real `article_prices.csv` only when the business data owner provides one.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

On first run, the app downloads configured Drive files into `data/`. Do not commit downloaded datasets, images or model artifacts; `.gitignore` excludes them.

## Streamlit Cloud

1. Push this repository to GitHub.
2. Create a Streamlit app with `app.py` as the main file.
3. Use Python 3.11 if available.
4. Add the dependencies from `requirements.txt`.
5. Ensure the deployed environment can reach the shared Google Drive files.
6. Add the real model and feature file IDs to `config/data_sources.json` only after they are available.

For large model/feature files, use an approved external artifact store or a secure deployment secret rather than GitHub history. Never commit the full H&M image dataset or customer-level files to a public repository.

## Pages

- **Executive Overview:** thesis KPIs and relative demand-versus-catalogue representation.
- **Customer Intelligence:** customer intention distribution, confidence and cold-start visibility.
- **Intention Analytics:** intention-level product explorer with product cards and image fallback status.
- **Product & Assortment:** catalogue intelligence and product metadata explorer.
- **AI Personalised Recommendation:** exact Three-Tower artifact gate; no fallback ranking is used.
- **Model Performance:** supplied offline model comparison and business interpretation.

## Important interpretation limits

Supply–demand gap is a relative comparison of customer intention share and catalogue share in the analytical sample. It is not an inventory shortage, revenue forecast or causal business effect. Offline AUC improvement is not evidence of revenue uplift without a future online experiment.

## Repository structure

```text
fashion-intention-bi/
├── app.py
├── pages/
├── src/
│   ├── data_loader.py
│   ├── customer_engine.py
│   ├── intention_engine.py
│   ├── recommendation_engine.py
│   └── ui.py
├── config/data_sources.json
├── requirements.txt
├── .streamlit/config.toml
└── README.md
```
