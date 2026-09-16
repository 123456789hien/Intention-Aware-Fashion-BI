from __future__ import annotations
import pandas as pd


def taxonomy(summary: dict) -> pd.DataFrame:
    rows = summary.get("intentions", {}).get("sampling_stats", [])
    return pd.DataFrame(rows).rename(columns={"intention_id": "intention_id"})


def intention_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("intention_") and c[10:].isdigit()]


def add_intention_label(df: pd.DataFrame, tax: pd.DataFrame, id_col: str = "dominant_intention") -> pd.DataFrame:
    out = df.copy()
    labels = dict(zip(tax.intention_id.astype(str), tax.intention_name)) if not tax.empty else {}
    out["intention_name"] = out[id_col].astype(str).map(labels).fillna("Unknown intention")
    return out


def catalog_supply(catalog: pd.DataFrame, tax: pd.DataFrame) -> pd.DataFrame:
    counts = catalog.groupby("dominant_intention", dropna=False).size().rename("catalogue_products").reset_index()
    counts["intention_id"] = counts["dominant_intention"].astype(int)
    return counts.merge(tax[["intention_id", "intention_name"]], on="intention_id", how="left")


def demand_supply_gap(user_dist: pd.DataFrame, catalog: pd.DataFrame, tax: pd.DataFrame) -> pd.DataFrame:
    demand = user_dist.rename(columns={"topic": "intention_id", "user_share": "demand_share"})[["intention_id", "intention_name", "demand_share"]]
    supply = catalog_supply(catalog, tax)
    supply["supply_share"] = supply["catalogue_products"] / supply["catalogue_products"].sum()
    out = demand.merge(supply[["intention_id", "catalogue_products", "supply_share"]], on="intention_id", how="outer")
    out["gap_pp"] = (out["demand_share"].fillna(0) - out["supply_share"].fillna(0)) * 100
    out["signal"] = out["gap_pp"].map(lambda x: "Potential under-representation" if x > 0 else "Potential over-representation")
    return out.sort_values("gap_pp", ascending=False)
