# How to contribute an A2 experiment

PRs are welcome for data quality, tests, case audits and strategy hypotheses. A proposal should be falsifiable, not merely a better in-sample headline.

1. **Hypothesis**: e.g. "a two-session rebound after a high-volume −3% close helps distinguish temporary shakeouts from genuine downtrends".
2. **Baseline & timing**: A2 V1.0 unchanged; all predicates use completed bars; next-regular-session open for any fill.
3. **Data**: specify period, split/dividend basis, source provenance, daily volume denominator and missing-bar handling. Do not commit raw third-party data without explicit redistribution rights.
4. **Comparison**: include per-trade positive *and* negative deltas, MDD, CAGR, Payoff, profit factor, win rate, trade count, market exposure, time out of market and plausible slippage/fees.
5. **Guardrails**: explicitly inspect the 2003 N20 missed winner, 2020 fake death cross and June shock, 2001/2004/2008/2015/2022 stop saves; they are diagnostic, not an adequate substitute for out-of-sample validation.
6. **Reproduction**: add small deterministic synthetic tests and a command that produces output from users' own CSVs; avoid committing confidential prices and broker credentials.
7. **Versions**: put experimental code under `research/experiments/`, results under `research/results/`, and document status in `docs/EXPERIMENTS.md`; no breaking changes to V1.0 without an explicitly reviewed V2 major proposal.

Before submitting:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

**Review outcomes**: `exploration`, `rejected`, `shadow-track`, or `validated-with-external-data`. Do not call an experiment validated solely because one historical sample improves its metrics.
