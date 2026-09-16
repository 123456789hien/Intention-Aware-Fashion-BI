import json
from pathlib import Path
import pandas as pd
from src.intention_engine import demand_supply_gap, taxonomy

ROOT = Path(__file__).resolve().parents[1]

def test_sources_config_is_valid_json():
    cfg = json.loads((ROOT / "config/data_sources.json").read_text())
    assert cfg["policy"]["no_simulated_data"] is True
    assert cfg["sources"]["article_metadata"]["file_id"]

def test_gap_is_derived_from_supplied_shares():
    summary = {"intentions": {"sampling_stats": [{"intention_id": 0, "intention_name": "A"}, {"intention_id": 1, "intention_name": "B"}]}}
    users = pd.DataFrame({"topic": [0,1], "intention_name": ["A","B"], "user_share": [0.75,0.25]})
    catalog = pd.DataFrame({"dominant_intention": [0,1,1,1]})
    gap = demand_supply_gap(users, catalog, taxonomy(summary))
    row = gap[gap.intention_id == 0].iloc[0]
    assert round(row.gap_pp, 2) == 50.0
