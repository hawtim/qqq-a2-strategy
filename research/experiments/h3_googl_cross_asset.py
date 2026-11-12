"""Cross-asset H3 research on Alphabet Class A (GOOGL), preserving frozen H3.

Uses the *unchanged* H3 engine and parameterization:
  a2_v3_1_h3_trend_spread_quick_reentry.replay(signals, "H3_FROZEN")
No experimental pullback selling/re-entry overrides are applied.

Downloads data for this research run, validates continuity / corporate actions,
and writes DERIVED trades/equity/summary only. NO raw third-party OHLCV exports.

Usage:
  python research/experiments/h3_googl_cross_asset.py \
      --end 2026-11-11 --outdir research/results/h3_googl_local
Run in a network-connected environment. The script is reproducible, but
provider snapshots may later revise adjustment factors and historical bars.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import urlencode

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "experiments"))

from a2_ema_dynamic_n20 import completed_month_signals
from a2_month_ema5_slope_regime import with_month_slope_features
from a2_v3_1_h3_trend_spread_quick_reentry import replay
from a2_v3_selected import replay_selected_v3
from qqq_a2.engine import Config, load_adjusted_csv


START = "2004-08-19"
DEFAULT_END = "2026-11-11"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) H3 academic historical research",
    "Accept": "application/json,text/csv,*/*",
}


def get_bytes(url: str) -> bytes:
    with urlopen(Request(url, headers=HEADERS), timeout=40) as response:
        payload = response.read()
    if not payload or len(payload) < 200:
        raise ValueError("Empty or insufficient provider response")
    return payload


def from_yahoo(end: str) -> tuple[pd.DataFrame, dict]:
    start_ts = int(datetime(2004, 8, 18, tzinfo=timezone.utc).timestamp())
    end_ts = int(pd.Timestamp(end, tz="UTC").timestamp()) + 86400
    url = "https://query1.finance.yahoo.com/v8/finance/chart/GOOGL?" + urlencode({
        "period1": start_ts, "period2": end_ts, "interval": "1d",
        "events": "div,splits", "includeAdjustedClose": "true",
    })
    payload = get_bytes(url)
    result = json.loads(payload)["chart"]["result"][0]
    indicator = result["indicators"]
    quote = indicator["quote"][0]
    adj = indicator["adjclose"][0]["adjclose"]
    df = pd.DataFrame({
        "Date": [datetime.fromtimestamp(t, timezone.utc).date()
                 for t in result["timestamp"]],
        "Open": quote["open"], "High": quote["high"], "Low": quote["low"],
        "Close": quote["close"], "Adj Close": adj,
        "Volume": quote["volume"],
    })
    return df, {"provider": "Yahoo Finance chart API", "url": url,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "basis": "Yahoo split-adjusted OHLC; Adj Close includes dividends"}


def from_stooq(end: str) -> tuple[pd.DataFrame, dict]:
    url = "https://stooq.com/q/d/l/?" + urlencode({
        "s": "googl.us", "i": "d", "d1": "20040819",
        "d2": end.replace("-", ""),
    })
    payload = get_bytes(url)
    df = pd.read_csv(io.BytesIO(payload))
    if not {"Date", "Open", "High", "Low", "Close", "Volume"} <= set(df.columns):
        raise ValueError("Unexpected Stooq CSV; possible blocked download")
    # Stooq may adjust OHLC for corporate actions but has no separately
    # disclosed dividend-adjusted close. Explicitly use a price basis proxy.
    df["Adj Close"] = df["Close"]
    return df, {"provider": "Stooq historical GOOGL.US",
                "url": url, "sha256": hashlib.sha256(payload).hexdigest(),
                "basis": "provider-supplied split/corporate-action adjusted OHLC; dividend proxy unverified"}


def from_github_snapshot(end: str) -> tuple[pd.DataFrame, dict]:
    url = "https://raw.githubusercontent.com/wellisonS/google-stock/main/googl_daily_prices.csv"
    payload = get_bytes(url)
    f = pd.read_csv(io.BytesIO(payload))
    f = f.rename(columns={
        "date": "Date", "1. open": "Open", "2. high": "High",
        "3. low": "Low", "4. close": "Close", "5. volume": "Volume",
    })
    f["Date"] = pd.to_datetime(f.Date)
    # This file contains as-traded pre-2022 prices and no adjusted close.
    # Handle both declared split events rather than treating splits as crashes.
    # 2014-04-03 class A rename & Class C distribution: historical prices /2.
    # 2022-07-18 20-for-1: prices /20.
    split = pd.Series(1., index=f.index)
    split.loc[f.Date < pd.Timestamp("2022-07-18")] /= 20.
    split.loc[f.Date < pd.Timestamp("2014-04-03")] /= 2.
    for col in ["Open", "High", "Low", "Close"]:
        f[col] = f[col].astype(float) * split
    f["Adj Close"] = f["Close"]
    return f, {"provider": "wellisonS/google-stock GitHub snapshot",
               "url": url, "sha256": hashlib.sha256(payload).hexdigest(),
               "basis": "manually 2014/2022 split-adjusted OHLC; excludes cash-dividend reinvestment"}


def qa_data(frame: pd.DataFrame, end: str, provider: dict) -> pd.DataFrame:
    df = frame.copy()
    df.Date = pd.to_datetime(df.Date)
    df = df.loc[(df.Date >= START) & (df.Date <= end)].copy()
    df = df.sort_values("Date").reset_index(drop=True)
    if df.empty or len(df) < 4900 or df.Date.duplicated().any():
        raise ValueError("Too few bars, duplicates or empty data")
    cols = ["Open", "High", "Low", "Close", "Adj Close"]
    if df[cols].isna().any().any() or (df[cols] <= 0).any().any():
        raise ValueError("Missing / nonpositive bar; no imputation")
    if not ((df.High + 1e-6 >= df[["Open", "Close", "Low"]].max(axis=1)) &
            (df.Low - 1e-6 <= df[["Open", "Close", "High"]].min(axis=1))).all():
        raise ValueError("OHLC bounds inconsistent")
    if df.Date.iloc[0].strftime("%Y-%m-%d") != START:
        raise ValueError("Missing initial IPO bar")
    maxgap = df.Date.diff().dt.days.max()
    if maxgap > 8:
        raise ValueError(f"Unexplained interior trading gap >8 days: {maxgap}")
    if (df.Close.pct_change().abs() > .5).any():
        raise ValueError("Suspicious >50% single-day jump; check splits")
    provider["date_start"] = str(df.Date.iloc[0].date())
    provider["date_end"] = str(df.Date.iloc[-1].date())
    provider["rows"] = len(df)
    provider["max_calendar_gap_days"] = int(maxgap)
    provider["raw_last_close"] = float(df.Close.iloc[-1])
    return df


def load_prices(end: str) -> tuple[pd.DataFrame, dict]:
    errors = {}
    for loader in [from_yahoo, from_stooq, from_github_snapshot]:
        try:
            df, prov = loader(end)
            df = qa_data(df, end, prov)
            if df.Date.iloc[-1] < pd.Timestamp(end):
                prov["WARNING"] = ("Source is stale: not a complete " + end +
                                   " backtest; latest available bar used")
            return df, prov
        except Exception as exc:
            errors[loader.__name__] = repr(exc)
    raise RuntimeError("No complete and valid GOOGL source: " + json.dumps(errors))


def baseline_buy_hold(data: pd.DataFrame, config: Config) -> dict:
    start, end = data.iloc[0], data.iloc[-1]
    qty = config.initial_cash / (start.adj_open * (1 + config.fee_per_side))
    equity = qty * data.adj_close
    high = equity.cummax()
    years = (end.Date - start.Date).days / 365.25
    return {
        "end_equity": float(equity.iloc[-1]),
        "cagr": float((equity.iloc[-1] / config.initial_cash) ** (1 / years) - 1),
        "mdd": float((equity / high - 1).min()),
        "completed_trades": 0,
        "exposure": 1.,
        "notes": "IPO first regular opening fill; commission one side; no terminal sale",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--outdir", default="research/results/h3_googl_local")
    args = parser.parse_args()

    raw, provenance = load_prices(args.end)
    # Feed *normalized* split/dividend-adjusted OHLC to the unchanged engine.
    data = raw.copy()
    factor = data["Adj Close"].astype(float) / data["Close"].astype(float)
    for col in ["Open", "High", "Low", "Close"]:
        data["adj_" + col.lower()] = data[col].astype(float) * factor
    config = Config(start_date=START)
    signals = with_month_slope_features(completed_month_signals(data))
    h3 = replay(signals, "H3_FROZEN", config=config)
    v3 = replay_selected_v3(signals, config=config)
    bh = baseline_buy_hold(data, config)

    summary = pd.DataFrame([
        {"ticker": "GOOGL", "version": "H3_NEGATIVE_ACCELERATING_MONTH_EMA5_FAST",
         **h3["stats"]},
        {"ticker": "GOOGL", "version": "V3_AUXILIARY_CONTROL", **v3["stats"]},
        {"ticker": "GOOGL", "version": "BUY_AND_HOLD_IPO_OPEN",
         **bh},
    ])
    output = Path(args.outdir)
    output.mkdir(exist_ok=True, parents=True)
    summary.to_csv(output / "summary.csv", index=False)
    h3["trades"].to_csv(output / "h3_trades.csv", index=False)
    h3["equity"].to_csv(output / "h3_equity.csv", index=False)
    v3["trades"].to_csv(output / "v3_trades.csv", index=False)
    manifest = {
        "symbol": "GOOGL", "security": "Alphabet Class A",
        "old_symbol": "GOOG (same Class A before 2014-04-03)",
        "requested_end": args.end, "actual_end": provenance["date_end"],
        "data": provenance, "ticker_start": START,
        "strategy": "Unmodified H3_FROZEN / H3_NEGATIVE_ACCELERATING_MONTH_EMA5_FAST",
        "code": "research/experiments/a2_v3_1_h3_trend_spread_quick_reentry.py",
        "strategy_parameter_changes": {},
        "comparators": ["unmodified V3 selected", "IPO buy and hold"],
        "initial_cash_usd": 100000,
        "fees_each_side": .0005, "slippage": 0, "tax": 0,
        "execution": "completed-close signal, next regular open fill",
        "cash_interest": 0,
        "past_month_known_only": True,
        "in_sample_cross_ticker_not_out_of_sample": True,
        "original_prices_not_exported": True,
        "warning": "One-name tests can contain survivorship and selection bias; do not extrapolate to other stocks",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False,
                                                  indent=2, default=str) + "\n")
    print("H3_GOOGLE_RESEARCH_DONE", json.dumps({
        "actual_end": provenance["date_end"],
        "rows": provenance["rows"], "provider": provenance["provider"],
        "h3": h3["stats"], "v3": v3["stats"], "buy_hold": bh,
        "trades_head": h3["trades"].head(3).to_dict(orient="records"),
        "trades_tail": h3["trades"].tail(5).to_dict(orient="records"),
    }, default=str, ensure_ascii=False))


if __name__ == "__main__":
    main()
