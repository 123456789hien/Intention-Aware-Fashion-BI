"""
trends.py
================================================================================
Turns the monthly revenue-by-segment table (Script 01 bonus step) into
month / quarter / year views, and computes period-over-period comparisons
("vs previous period" or "vs same period last year") — the two comparison
modes a real merchandising team actually uses.

All figures here trace back to real transaction dates (t_dat) grouped by a
product's dominant intention — not simulated data.
================================================================================
"""
from __future__ import annotations

import pandas as pd


def _month_to_quarter(ym: str) -> str:
    year, month = ym.split("-")
    q = (int(month) - 1) // 3 + 1
    return f"{year}-Q{q}"


def _month_to_year(ym: str) -> str:
    return ym.split("-")[0]


def with_period_columns(monthly_df: pd.DataFrame) -> pd.DataFrame:
    df = monthly_df.copy()
    if df.empty:
        df["quarter"] = []
        df["year"] = []
        return df
    df["quarter"] = df["year_month"].apply(_month_to_quarter)
    df["year"] = df["year_month"].apply(_month_to_year)
    return df


GRANULARITY_COL = {"Month": "year_month", "Quarter": "quarter", "Year": "year"}


def period_options(df: pd.DataFrame, granularity: str) -> list[str]:
    col = GRANULARITY_COL[granularity]
    if df.empty:
        return []
    return sorted(df[col].unique().tolist())


def aggregate_period(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    """Collapses the monthly table to the chosen granularity, recomputing
    revenue_share_pct within each period (not just summing the monthly
    shares, which would be mathematically wrong). granularity="All time"
    collapses across the ENTIRE date range instead of one period."""
    if granularity == "All time":
        return aggregate_all(df)
    col = GRANULARITY_COL[granularity]
    if df.empty:
        return df
    agg = df.groupby([col, "intention"]).agg(
        revenue=("revenue", "sum"),
        units=("units", "sum"),
        n_customers=("n_customers", "sum"),
        avg_price=("avg_price", "mean"),
    ).reset_index()
    totals = agg.groupby(col)["revenue"].transform("sum")
    agg["revenue_share_pct"] = (agg["revenue"] / totals * 100).round(4)
    return agg


def aggregate_all(df: pd.DataFrame) -> pd.DataFrame:
    """Collapses the ENTIRE monthly table (all 25 months) into one row per
    segment — the 'All time' view, for when the user wants the full-period
    picture instead of one month/quarter/year."""
    if df.empty:
        return df
    agg = df.groupby("intention").agg(
        revenue=("revenue", "sum"),
        units=("units", "sum"),
        n_customers=("n_customers", "sum"),
        avg_price=("avg_price", "mean"),
    ).reset_index()
    total = agg["revenue"].sum()
    agg["revenue_share_pct"] = (agg["revenue"] / total * 100).round(4)
    return agg


def _shift_period(period: str, granularity: str, mode: str) -> str | None:
    """Returns the label of the comparison period, or None if out of range
    or not applicable (e.g. 'same period last year' for Year granularity)."""
    try:
        if granularity == "Month":
            year, month = int(period[:4]), int(period[5:7])
            if mode == "Same period last year":
                year -= 1
            else:
                month -= 1
                if month == 0:
                    month, year = 12, year - 1
            return f"{year:04d}-{month:02d}"
        elif granularity == "Quarter":
            year, q = int(period[:4]), int(period[-1])
            if mode == "Same period last year":
                year -= 1
            else:
                q -= 1
                if q == 0:
                    q, year = 4, year - 1
            return f"{year}-Q{q}"
        else:  # Year
            return str(int(period) - 1)
    except (ValueError, IndexError):
        return None


def filter_by_period(df: pd.DataFrame, date_col: str, granularity: str, period) -> pd.DataFrame:
    """Filters any transaction-level DataFrame (must have a YYYY-MM-DD date
    column) down to the chosen reporting period. granularity="All time" (or
    period=None) returns the unfiltered df — used by pages that reuse the
    shared sidebar period selector but apply it to raw transactions rather
    than the pre-aggregated monthly_segment_trends.csv."""
    if granularity == "All time" or period is None or df.empty:
        return df
    ym = df[date_col].astype(str).str[:7]
    if granularity == "Month":
        return df[ym == period]
    elif granularity == "Quarter":
        return df[ym.apply(_month_to_quarter) == period]
    else:  # Year
        return df[ym.apply(_month_to_year) == period]


def period_to_date_range(granularity: str, period: str):
    """Converts a (granularity, period) selection into a real [start, end]
    date range for filtering raw transaction rows (t_dat). Returns
    (None, None) for "All time" or when no period is selected — callers
    should treat that as "no date filter, use everything"."""
    if granularity == "All time" or period is None:
        return None, None
    try:
        if granularity == "Month":
            start = pd.Timestamp(f"{period}-01")
            end = start + pd.offsets.MonthEnd(0)
        elif granularity == "Quarter":
            year, q = int(period[:4]), int(period[-1])
            start_month = (q - 1) * 3 + 1
            start = pd.Timestamp(year=year, month=start_month, day=1)
            end = start + pd.offsets.QuarterEnd(0)
        else:  # Year
            year = int(period)
            start = pd.Timestamp(year=year, month=1, day=1)
            end = pd.Timestamp(year=year, month=12, day=31)
        return start, end
    except (ValueError, IndexError):
        return None, None


def segment_comparison(agg_df: pd.DataFrame, period: str, granularity: str, mode: str) -> pd.DataFrame:
    """
    For each of the 10 segments, returns current-period revenue_share_pct,
    the comparison-period value (NaN if unavailable), and the delta in
    percentage points — the number shown as "↑ +2.1pp" on segment cards.
    granularity="All time" has no period column and no prior period —
    every segment's row is just its all-time total, delta_pp is always None.
    """
    if granularity == "All time":
        current = agg_df.set_index("intention")
        rows = []
        for k in range(10):
            cur_share = float(current.loc[k, "revenue_share_pct"]) if k in current.index else None
            cur_revenue = float(current.loc[k, "revenue"]) if k in current.index else None
            rows.append({
                "intention": k, "revenue_share_pct": cur_share, "revenue": cur_revenue,
                "prior_share_pct": None, "delta_pp": None, "prior_label": None,
            })
        return pd.DataFrame(rows).set_index("intention")

    col = GRANULARITY_COL[granularity]
    current = agg_df[agg_df[col] == period].set_index("intention")

    prior_label = _shift_period(period, granularity, mode)
    if prior_label is not None and prior_label in agg_df[col].values:
        prior = agg_df[agg_df[col] == prior_label].set_index("intention")
    else:
        prior = pd.DataFrame(columns=agg_df.columns).set_index("intention")

    rows = []
    for k in range(10):
        cur_share = float(current.loc[k, "revenue_share_pct"]) if k in current.index else None
        cur_revenue = float(current.loc[k, "revenue"]) if k in current.index else None
        prior_share = float(prior.loc[k, "revenue_share_pct"]) if k in prior.index else None
        delta = (cur_share - prior_share) if (cur_share is not None and prior_share is not None) else None
        rows.append({
            "intention": k, "revenue_share_pct": cur_share, "revenue": cur_revenue,
            "prior_share_pct": prior_share, "delta_pp": delta, "prior_label": prior_label,
        })
    return pd.DataFrame(rows).set_index("intention")
