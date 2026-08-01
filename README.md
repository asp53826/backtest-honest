# backtest-honest

A backtester built to measure its own selection bias, calibrated on markets
where the correct answer is known to be **nothing**.

[![CI](https://github.com/asp53826/backtest-honest/actions/workflows/ci.yml/badge.svg)](https://github.com/asp53826/backtest-honest/actions/workflows/ci.yml)
![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776AB?style=flat-square)
![runtime deps](https://img.shields.io/badge/runtime_dependencies-1-2ea44f?style=flat-square)

> Search 500 moving-average variants on a pure random walk and the best one
> posts a Sharpe of **0.99**. Nothing was learned — returns were independent
> draws. The number is the maximum of 500 nulls, and a backtest with no
> correction reports it as a discovery **7.5%** of the time.

## The oracle is a market with no signal

Most backtesting libraries can only tell you a strategy did well. That is
unfalsifiable: there is no baseline for how well a *worthless* strategy should
have done. So every calibration here runs on random walks — i.i.d. returns, no
momentum, no mean reversion — where the true Sharpe of every strategy is exactly
zero and any measured performance is selection bias made visible.

The generator is checked in both directions: the null has autocorrelation
`|ρ| < 0.03`, and the alternative genuinely has the momentum it claims.

## Measured, not implied

### Alpha appears from searching alone

2,000 bars of random walk. Every row is the same market; only the number of
variants tried changes.

| variants | best Sharpe | theoretical max | deflated p | verdict |
|---:|---:|---:|---:|---|
| 1 | 0.605 | 0.000 | 0.956 | *SKILL* |
| 5 | 0.985 | 0.424 | 0.943 | search |
| 25 | 0.985 | 0.709 | 0.782 | search |
| 100 | 0.985 | 0.899 | 0.596 | search |
| 500 | 0.985 | 1.084 | 0.391 | search |
| 2,000 | 1.103 | 1.224 | 0.367 | search |
| 7,021 | 1.103 | 1.340 | 0.253 | search |

Best-in-sample Sharpe climbs with the size of the search and tracks the
theoretical maximum of that many null draws — `E[max]` grows like `√(2 log N)`,
so it never stops rising and never needs a real signal.

**The first row is a false positive and is left in.** At one variant there is no
multiple testing to correct, so the test reduces to a Sharpe against zero and
calls 0.605 over 2,000 bars significant. The correction addresses selection
bias, not the ordinary error of a single estimate, and it is not advertised as
doing both.

### False discovery rate

200 independent random-walk markets, a 500-variant search on each. True
discoveries available: **zero**.

| method | discoveries | rate |
|---|---:|---:|
| naive — best Sharpe > 1.0 | 15 | **7.5%** |
| deflated Sharpe p > 0.95 | 0 | **0.0%** |

### What it costs: the correction has low power

A correction that rejects everything is not a correction, so the same
500-variant search runs on markets with genuine AR(1) momentum.

| bars | φ | best Sharpe | deflated p | verdict |
|---:|---:|---:|---:|---|
| 2,000 | 0.00 | 0.985 | 0.391 | search |
| 2,000 | 0.35 | 2.207 | 0.999 | **SKILL** |
| 5,000 | 0.20 | 0.727 | 0.574 | search |
| 20,000 | 0.10 | 0.289 | 0.317 | search |
| 20,000 | 0.20 | 0.505 | 0.925 | search |
| 20,000 | 0.35 | 1.234 | 1.000 | **SKILL** |

It discriminates — φ = 0.35 is accepted at every sample size, the null rejected
at every sample size. But the bar is high. **φ = 0.10, a real and exploitable
edge, is never accepted — not even with 20,000 bars.** More data does not fix
it, because searching 500 variants raises the bar faster than the extra data
lowers it. That is the trade, stated plainly: false positives bought down with
false negatives.

### Leakage from overlapping labels

Features built only from past draws, labels only from future draws — no genuine
relationship exists, so the correct score is zero. 60 seeds, ± is the standard
error of the mean.

| label horizon | plain k-fold | purged | purged + embargo | train dropped |
|---:|---:|---:|---:|---:|
| 1 | 0.0003 ±0.0052 | 0.0004 ±0.0052 | 0.0003 ±0.0051 | 2.2% |
| 20 | −0.0018 ±0.0106 | −0.0059 ±0.0110 | −0.0091 ±0.0112 | 5.3% |
| 50 | 0.0094 ±0.0101 | 0.0033 ±0.0104 | 0.0046 ±0.0105 | 10.3% |
| 100 | **0.0322 ±0.0116** | 0.0072 ±0.0111 | 0.0053 ±0.0117 | 18.7% |

At horizon 100 plain k-fold reports skill 2.8 standard errors from zero on data
containing none; purging brings it inside the noise. At horizon 1, where labels
do not overlap, the two are identical — the correction costs nothing when there
is nothing to correct, which is how you know it is doing the right thing.

**Leakage needs two ingredients, and an earlier version of this benchmark had
only one.** Overlapping labels are not enough: with i.i.d. features a
nearest-neighbour model picks temporally random training rows, never lands on an
overlapping one, and plain k-fold scores zero. Real financial features are
smoothed prices and moving averages, so temporal proximity *is* feature
proximity — that is what makes the overlap reachable. The first run measured
nothing and looked like a clean result.

## Where it loses

- **Low power against modest edges**, quantified above. After a 500-variant
  search a φ = 0.10 signal is indistinguishable from having looked hard enough.
  If your search is wide, this test will not certify anything short of a large
  effect.
- **No correction at one trial.** Deflated Sharpe reduces to a Sharpe against
  zero, and the table shows it producing a false positive on noise.
- **`E[max]` is a Gumbel asymptotic.** Validated against simulation — error
  0.057 at N = 2 falling to 0.006 by N = 5,000 — so it is least accurate exactly
  where it matters least.
- **The variant count must be honest.** Every number here assumes you report how
  many strategies you actually tried, including the ones abandoned before they
  were written down. Nothing in the code can check that.
- **Synthetic markets only.** Random walks and AR(1) have no regime changes, no
  fat tails, no structural breaks. The calibration transfers; the specific
  Sharpe numbers do not.
- **Purging is measured by rows dropped, not by accuracy lost.** At a 200-bar
  horizon on 1,200 bars it removes 33% of training data; whether the remaining
  data is *enough* is a separate question this repo does not answer.

## Verify it

```bash
make test      # 23 tests
make overfit   # alpha from searching, and the power analysis
make leakage   # overlapping labels, purging, and what it costs
```

## Use it

```python
from bt.stats import deflated_sharpe, expected_max_sharpe, sharpe
from bt.cv import PurgedKFold, forward_label_spans

sharpe(returns)                          # annualised, zero risk-free
expected_max_sharpe(n_trials=500)        # what noise alone would give you
deflated_sharpe(returns, n_trials=500)   # P(skill), not P(positive)

spans = forward_label_spans(n, horizon=20)
for train, test in PurgedKFold(5, embargo_pct=0.02).split(n, spans):
    ...                                  # train never shares a bar with test
```

`deflated_sharpe` takes `n_trials` as a required argument on purpose. A Sharpe
without the size of the search that produced it is not interpretable.

## License

MIT
