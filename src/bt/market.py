"""Price series with a known edge, including an edge of exactly zero.

The null here is not "a strategy that did badly" — it is a market in which no
strategy can do well, because the returns are independent draws. Anything a
search finds on `random_walk` is measured selection bias, and that is the only
way to calibrate what a backtest result is worth.

`with_drift` adds a real, known signal so the same machinery can be checked in
the other direction: a correction that rejects everything is not a correction,
it is a broken test.
"""

from __future__ import annotations

import numpy as np


def random_walk(n: int, seed: int = 0, vol: float = 0.01) -> np.ndarray:
    """Log-price random walk. Returns are i.i.d. — no autocorrelation, no
    momentum, no mean reversion, nothing to find."""
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0, vol, size=n)
    return np.exp(np.cumsum(r))


def with_momentum(n: int, seed: int = 0, vol: float = 0.01,
                  phi: float = 0.10) -> np.ndarray:
    """A walk with genuine AR(1) momentum — a real, exploitable edge.

    `phi` is the autocorrelation of returns. At 0.10 the signal is small enough
    to be realistic and large enough that a correctly-run backtest should find
    it, which is what distinguishes a useful correction from a paranoid one.
    """
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, vol, size=n)
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = phi * r[t - 1] + eps[t]
    return np.exp(np.cumsum(r))


def returns(prices: np.ndarray) -> np.ndarray:
    return np.diff(np.log(np.asarray(prices, dtype=float)))


# ---------------------------------------------------------------------------
# a strategy universe large enough to overfit with
# ---------------------------------------------------------------------------

def ma_crossover(prices: np.ndarray, fast: int, slow: int) -> np.ndarray:
    """Position in {-1, +1} from a moving-average crossover.

    Signals are shifted one bar so a position is taken *after* the bar that
    produced it. Without that shift the backtest trades on the close it is
    predicting, which manufactures an edge out of nothing and is the single most
    common backtest bug.
    """
    if fast >= slow:
        raise ValueError("fast window must be shorter than slow")
    p = np.asarray(prices, dtype=float)
    f = _sma(p, fast)
    s = _sma(p, slow)
    sig = np.where(f > s, 1.0, -1.0)
    return np.concatenate([[0.0], sig[:-1]])          # lag by one bar


def _sma(x: np.ndarray, w: int) -> np.ndarray:
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.empty_like(x)
    out[:w] = x[:w].mean() if w > 0 else x[0]
    out[w:] = (c[w + 1:] - c[1:-w]) / w
    return out


def strategy_returns(prices: np.ndarray, position: np.ndarray,
                     cost_bps: float = 0.0) -> np.ndarray:
    """Per-bar strategy return, optionally net of turnover cost.

    Costs default to zero because the benchmark's null is about *statistics*,
    not about whether a strategy survives frictions — but they are available so
    the cost of the search is not confused with the cost of trading.
    """
    r = returns(prices)
    pos = np.asarray(position, dtype=float)[:len(r)]
    gross = pos * r
    if cost_bps:
        turnover = np.abs(np.diff(np.concatenate([[0.0], pos])))
        gross = gross - turnover * (cost_bps / 1e4)
    return gross


def universe(max_slow: int = 120) -> list[tuple[int, int]]:
    """Every (fast, slow) pair up to `max_slow`.

    Deliberately a real search space rather than a handful of picks: at
    max_slow=120 there are thousands of variants, and the number of variants is
    the parameter the whole repo is about.
    """
    return [(f, s) for s in range(3, max_slow + 1) for f in range(2, s)]
