# AGENTS.md — rules for AI contributors and future strategy agents

## Purpose

This is a reproducible research repository for QQQ A2-N20. The task is not to maximize a historical backtest score. Preserve falsifiability, measured risk, and exact execution timing.

## Canonical baseline

1. `docs/STRATEGY_V1.md` and `src/qqq_a2/engine.py` define **A2-N20 V1.0**.
2. Baseline: fresh SMA20/60 gold cross after completed daily close, buy at next regular open; first 20 full trading days after fill only, fresh SMA10/20 death cross with close below entry adjusted open, exit next regular open; SMA20/60 fresh death cross exits regardless of age; no reentry until the next fresh SMA20/60 gold cross.
3. Frozen benchmark: strict natural start (no synthetic initial holding), 58 completed trades, 24 winners, 34 losers, 25 N20 exits, Payoff ≈4.02, MDD ≈−20.29%, 2026-07-13 cutoff.
4. Do not quietly change parameters, initialization, entry price basis, adjustment method, fees, fills, or data period. An alternative must have a distinct version and comparison report.

## Data and honesty

- Check `docs/DATA_SOURCES.md` before using data; do not add copyrighted/unlicensed price histories, paid FirstRate Data raw bars, secrets, brokerage tokens, or personal account exports.
- Signals may use only bars available at or before the completed session's close. No future label/maximum/lowest point may enter a strategy predicate.
- Entry and exit execution are next-session open. Include gap risks and fees; no fictional same-close fills or guaranteed stop prices.
- Historical results are sample-in and should not be described as out-of-sample edge or a guaranteed maximum drawdown.
- Different Payoff conventions matter: 58 completed-only = ~4.02; common-cutoff liquidation of final open trade = ~3.99. Never silently mix them.

## Research and contribution workflow

1. Read the relevant report in `docs/research/` and its source script in `research/experiments/`.
2. State the hypothesis, exact event timing and economic reason before testing. Freeze the primary metric and negative controls.
3. Keep A2 V1.0 unchanged; implement candidate in a separate experimental function/module.
4. Re-run `python -m unittest discover -s tests -v` and validate the 58-trade strict baseline when licensed local sources are available.
5. Report all changed trades, saved losers, lost winners, effect on account MDD, CAGR, trading costs, time in market, and subperiod sensitivity.
6. Do not optimize by repeatedly choosing lucky MA/ATR/volume numbers. Save disconfirming cases and failed experiments.

## Agent and trading boundaries

- Automation is **read-only by default**: data update, validation, state computation, alert generation and audit log.
- Never route orders, access broker keys, authorize leverage, or change the strategy under this repository's default permissions. Those require explicit user authorization and a separate execution system with dry-run and risk gates.
- On incomplete/latest data, report `STALE_DATA` or `UNKNOWN_STATE`; do not guess that the strategy is flat or long.
- Use `data_asof` in US/Eastern for the completed daily bar; separate `signal_date` from `execution_date`.
- Public reporting must not expose personal balances or positions.
