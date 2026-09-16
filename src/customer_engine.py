from __future__ import annotations
import pandas as pd


def customer_ids(weights: pd.DataFrame) -> list[str]:
    return weights["customer_id"].astype(str).tolist()


def profile(weights: pd.DataFrame, confidence: pd.DataFrame, customer_id: str) -> pd.Series:
    row = weights.loc[weights.customer_id.astype(str) == str(customer_id)]
    if row.empty: raise KeyError(f"Customer not found in supplied Drive data: {customer_id}")
    row = row.iloc[0].copy()
    conf = confidence.loc[confidence.customer_id.astype(str) == str(customer_id)]
    if not conf.empty:
        for c in ["n_purchases", "confidence", "is_cold_start", "dominant_intention_name"]:
            if c in conf.columns: row[c] = conf.iloc[0][c]
    return row


def top_intentions(row: pd.Series, labels: dict[int, str], n: int = 10) -> pd.DataFrame:
    cols = [c for c in row.index if c.startswith("intention_") and c[10:].isdigit()]
    out = pd.DataFrame({"intention_id": [int(c[10:]) for c in cols], "weight": [float(row[c]) for c in cols]})
    out["intention_name"] = out.intention_id.map(labels).fillna("Unknown intention")
    return out.sort_values("weight", ascending=False).head(n)
