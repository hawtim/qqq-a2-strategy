# AGENTS.md — rules for AI contributors and future strategy agents

## Purpose

This is a reproducible research repository for QQQ A2-N20. The task is not to maximize a historical backtest score. Preserve falsifiability, measured risk, and exact execution timing.

## Canonical baseline

1. `docs/STRATEGY_V1.md` and `src/qqq_a2/engine.py` define **A2-N20 V1.0**.
2. Baseline: fresh SMA20/60 gold cross after completed daily close, buy at next regular open; first 20 full trading days after fill only, fresh SMA10/20 death cross with close below entry adjusted open, exit next regular open; SMA20/60 fresh death cross exits regardless of age; no reentry until the next fresh SMA20/60 gold cross.
3. Frozen benchmark: strict natural start (no synthetic initial holding), 58 completed trades, 24 winners, 34 losers, 25 N20 exits, Payoff ≈4.02, MDD ≈−20.29%, 2026-07-13 cutoff.
4. Do not quietly change parameters, initialization, entry price basis, adjustment method, fees, fills, or data period. An alternative must have a distinct version and comparison report.

## Current selected research/paper baseline (2026-11-11)

1. **EMA V3 `EMA_V3_ADAPTIVE_N20_DAY0_CLOSE3` is the user-selected current research baseline**, not a replacement for the independently frozen SMA V1.0 benchmark. The single versioned entrypoint is `research/experiments/a2_v3_selected.py` → `replay_selected_v3()`; full exact rules live in `docs/STRATEGY_SELECTED_V3.md`.
2. V3 = archived V2 (`up_slow_veto_1_abs` plus deferred regular entry, FAST recovery and N20 on all entry days) **plus** two narrowly specified new exits: at regular-entry age 20, conditionally extend N20 to age 30 if 2/5 recent closes are below entry and EMA10/20 relative spread contracted over five sessions; at entry-day close, if gross price loss >=3%, exit next regular open. Original net 7% close stop and slow EMA20/60 death exit remain.
3. Version V2, previous EMA fourth option, SMA V1, QQQ buy-and-hold and original result files must all remain reproducible and separately labeled. Never silently re-label a V2 result as V3.
4. The historically better-performing global **4% closing stop is a negative control / in-sample parameter-selection-bias example** (`stop4_control`), **not the selected rule**. Do not confuse better historical return with proof of overfitting either: the problem is lack of independent validation. Preserve both the attractive figures and the disconfirming cases.
5. V3 and V2 are still research-only. There is **no brokerage order authorization**; all trades are completed-close signals and next-session regular-open fills, subject to unmodeled execution slippage. Do not classify V3 as proven out-of-sample or imply it guarantees a maximum 3%/7% loss.

## Frozen research candidate: QQQ H3 Profit Guard A/B v1.0 (2026-11-13)

- **Do not merge these into H3 by implication:** A/B are **research-only, separately frozen overlay candidates**, not a replacement of the canonical A2/V3/V3.1-H3 base rules or authorization to trade.
- Canonical human contract: `docs/STRATEGY_H3_PROFIT_GUARD_AB_V1.md`. Machine parameters: `research/config/h3_profit_guard_ab_v1.json`. Reproducible fixed entrypoint: `research/experiments/h3_profit_guard_ab_v1_frozen.py`.
- For both QQQ candidates, use as-of peak close profit >=10%, positive as-of current close profit, EMA20>EMA60, **no minimum holding-age gate (age0)**, at most **one extra sell and one special rebuy** per original H3 position. A extra exit: fresh EMA5/20 death cross. B: A plus EMA10>EMA20 and decline over last three completed closes >=1.5 ATR20. Special reentry: fresh EMA5/10 gold cross while EMA20>EMA60 and original H3 shadow still long; fill next regular open.
- H3 shadow original entry/exit, N20, monthly veto, FAST status must **not be reset** by intermediate real portfolio liquidation. Old age10 research results can be used for historical comparisons because age0 and age10 produce identical executions on the 2000–2026 QQQ dataset, but age0 is canonical prospectively.
- No silent revisions to A/B parameters. New age thresholds, multiple cycles, SPY/Nikkei, dynamic regimes or custom reentries are independent later versioned experiments. No claim of sample-out performance.

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
