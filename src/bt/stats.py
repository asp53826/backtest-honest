"""Sharpe ratios, and the corrections that make them mean something.

A Sharpe ratio computed once on one strategy is an estimate with a standard
error. A Sharpe ratio reported as the best of two hundred attempts is not an
estimate of anything — it is the maximum of two hundred draws, and its
expectation is positive even when every strategy is worthless.

Both corrections here are from Bailey and López de Prado. They are implemented
because the benchmark needs something to test the null against, and the null in
this repo is data with no signal in it whatsoever.
"""

from __future__ import annotations

import math

import numpy as np

SQRT_2PI = math.sqrt(2.0 * math.pi)
EULER = 0.5772156649015329


def sharpe(returns: np.ndarray, periods: int = 252) -> float:
    """Annualised Sharpe. Zero risk-free rate — the benchmark compares
    strategies against each other and against noise, never against cash."""
    r = np.asarray(returns, dtype=float)
    if r.size < 2:
        return 0.0
    sd = r.std(ddof=1)
    if sd < 1e-15:
        return 0.0
    return float(r.mean() / sd * math.sqrt(periods))


def sharpe_stderr(returns: np.ndarray, periods: int = 252) -> float:
    """Standard error of the Sharpe estimate, skew- and kurtosis-aware.

    The textbook 1/sqrt(n) understates it for the skewed, fat-tailed returns
    every real strategy produces. Lo (2002); the higher-moment terms are what
    make it honest on non-normal data.
    """
    r = np.asarray(returns, dtype=float)
    n = r.size
    if n < 3:
        return float("inf")
    sr = sharpe(r, periods=1)          # per-period, corrected at the end
    g1, g2 = _skew(r), _kurt(r)
    var = (1 - g1 * sr + (g2 - 1) / 4 * sr ** 2) / (n - 1)
    return float(math.sqrt(max(var, 0.0)) * math.sqrt(periods))


def _skew(r: np.ndarray) -> float:
    d = r - r.mean()
    s = r.std(ddof=0)
    return float((d ** 3).mean() / s ** 3) if s > 1e-15 else 0.0


def _kurt(r: np.ndarray) -> float:
    d = r - r.mean()
    s = r.std(ddof=0)
    return float((d ** 4).mean() / s ** 4) if s > 1e-15 else 3.0


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse normal CDF, Acklam's rational approximation (~1e-9)."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0,1), got {p}")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def expected_max_sharpe(n_trials: int, sharpe_var: float = 1.0) -> float:
    """Expected maximum Sharpe from `n_trials` strategies that have **no edge**.

    This is the number the whole repo exists to make visible. Search enough
    worthless strategies and the best one looks good — not because anything was
    learned but because the maximum of many draws is large. Growth is roughly
    sqrt(2 log N), so it never stops rising and never needs a real signal.
    """
    if n_trials < 2:
        return 0.0
    n = float(n_trials)
    z1 = _norm_ppf(1 - 1.0 / n)
    z2 = _norm_ppf(1 - 1.0 / (n * math.e))
    return float(math.sqrt(sharpe_var) * ((1 - EULER) * z1 + EULER * z2))


def deflated_sharpe(returns: np.ndarray, n_trials: int,
                    sharpe_var: float | None = None,
                    periods: int = 252) -> float:
    """Probability the observed Sharpe reflects skill rather than search.

    The benchmark is not zero — it is the Sharpe you would expect from the best
    of `n_trials` worthless strategies. Testing against zero is what lets a
    backtest of 200 variants report a discovery every time.

    Returns a probability. Below ~0.95 the result is indistinguishable from
    having looked hard enough.
    """
    r = np.asarray(returns, dtype=float)
    n = r.size
    if n < 3:
        return 0.0

    sr = sharpe(r, periods=1)
    if sharpe_var is None:
        sharpe_var = 1.0 / max(n - 1, 1)
    sr0 = expected_max_sharpe(n_trials, sharpe_var)

    g1, g2 = _skew(r), _kurt(r)
    denom = math.sqrt(max(1 - g1 * sr + (g2 - 1) / 4 * sr ** 2, 1e-12))
    z = (sr - sr0) * math.sqrt(max(n - 1, 1)) / denom
    return float(_norm_cdf(z))


def probabilistic_sharpe(returns: np.ndarray, benchmark_sr: float = 0.0,
                         periods: int = 252) -> float:
    """Deflated Sharpe against an arbitrary benchmark rather than the
    multiple-testing one. Same machinery, `n_trials = 1`."""
    r = np.asarray(returns, dtype=float)
    n = r.size
    if n < 3:
        return 0.0
    sr = sharpe(r, periods=1)
    sr0 = benchmark_sr / math.sqrt(periods)
    g1, g2 = _skew(r), _kurt(r)
    denom = math.sqrt(max(1 - g1 * sr + (g2 - 1) / 4 * sr ** 2, 1e-12))
    return float(_norm_cdf((sr - sr0) * math.sqrt(max(n - 1, 1)) / denom))
