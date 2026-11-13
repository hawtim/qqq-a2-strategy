"""Validate H3 cross-asset v0.1 RESEARCH schema; never place orders.

Lightweight contract test only. Does NOT backtest SPY, Nikkei or any
strategy, and does not make this design spec an executable trading engine.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
CFG=ROOT/"research/config/h3_cross_asset_research_framework_v0_1.json"
FROZEN=ROOT/"research/config/h3_profit_guard_ab_v1.json"
DOC=ROOT/"docs/research/H3_CROSS_ASSET_RESEARCH_SYSTEM_FRAMEWORK_V0_1.md"

def main():
    config=json.loads(CFG.read_text(encoding="utf-8"))
    frozen=json.loads(FROZEN.read_text(encoding="utf-8"))
    assert DOC.exists()
    assert config["schema_version"]=="0.1.0"
    assert config["type"]=="cross_asset_research_contract_not_executable_strategy"
    assert config["status"]=="DESIGN_SPEC_ONLY"
    assert config["invariants"]["execution"]=="next_regular_session_open"
    assert config["invariants"]["signal_generation"]=="completed_daily_close"
    assert config["invariants"]["no_live_execution_authorization"] is True
    assert config["reference_versions"]["profit_guard_frozen"]==(
        "research/config/h3_profit_guard_ab_v1.json")
    assert config["layer_status"]["core_trend"]["status"]=="FROZEN_REFERENCE"
    pg=config["layer_status"]["profit_guard"]
    assert pg["default"]=="NONE"
    assert pg["B_frozen"]["atr_window"]==20
    assert pg["B_frozen"]["atr_multiplier"]==1.5
    assert pg["B_frozen"]["cap_extra_sell"]==1
    assert pg["B_frozen"]["cap_special_rebuy"]==1
    assert pg["B_frozen"]["atr_window"] in pg["B_experiment_grid"]["atr_window"]
    assert pg["B_frozen"]["atr_multiplier"] in pg["B_experiment_grid"]["atr_multiplier"]
    assert set(pg["B_experiment_grid"]["atr_window"])=={14,20,30}
    assert set(pg["B_experiment_grid"]["atr_multiplier"])=={1.0,1.5,2.0}
    assert pg["B_sample_in_best_not_promoted"]["status"]=="RESEARCH_ONLY"
    assert config["layer_status"]["entry_shape"]["enabled_for_orders"] is False
    assert config["layer_status"]["volatility_regime"]["gate_override_enabled"] is False
    frozen_pg=next(v for v in frozen["variants"] if v["name"]=="B")
    assert frozen_pg["additional_conditions"]==[
        "EMA10[t]>EMA20[t]",
        "(AdjClose[t-3]-AdjClose[t])/ATR20[t]>=1.5"
    ]
    assert frozen_pg["max_cycles"]==1
    assert config["asset_profiles"][0]["asset_id"]=="QQQ"
    assets=[v["asset_id"] for v in config["asset_profiles"]]
    assert len(set(assets))==len(assets)
    assert "SPY" in assets and "NIKKEI225_PROXY_TBD" in assets
    for key in ["invariants","layer_status","experiment_rules"]:
        assert key in config
    for relative in [
        "docs/STRATEGY_H3_PROFIT_GUARD_AB_V1.md",
        "research/experiments/a2_v3_selected.py",
        "research/experiments/a2_v3_1_h3_trend_spread_quick_reentry.py",
    ]:
        assert (ROOT/relative).exists()
    print(json.dumps({
        "validation":"PASS",
        "framework":"H3 cross-asset v0.1 design spec",
        "frozen_guard_B_unchanged":True,
        "safe_to_trade":False,
        "asset_profiles":assets,
        "B_grid_candidates":(
            len(pg["B_experiment_grid"]["atr_window"])*
            len(pg["B_experiment_grid"]["atr_multiplier"]))
    },ensure_ascii=False))

if __name__=="__main__":
    main()
