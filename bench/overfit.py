"""Finding alpha in a market that has none.

    uv run python bench/overfit.py

Every price series here is a random walk. Returns are independent draws, so no
strategy can have an edge and every Sharpe reported below is selection bias
being measured directly.
"""

from __future__ import annotations

import math

import numpy as np

from bt.market import (ma_crossover, random_walk, strategy_returns, universe,
                       with_momentum)
from bt.stats import (deflated_sharpe, expected_max_sharpe, sharpe,
                      sharpe_stderr)


def hdr(t):
    print(f"\n=== {t} ===\n")


def search(prices, variants, cost_bps=0.0):
    """In-sample Sharpe of every variant; returns (best_params, best_sr, all)."""
    out = []
    for f, s in variants:
        r = strategy_returns(prices, ma_crossover(prices, f, s), cost_bps)
        out.append(sharpe(r))
    arr = np.asarray(out)
    i = int(np.argmax(arr))
    return variants[i], float(arr[i]), arr


def main():
    N_BARS = 2000

    hdr("the best of N worthless strategies, on a market with no signal")
    print("  Prices are a random walk: independent returns, no momentum, no")
    print("  mean reversion. The true Sharpe of every strategy below is zero.\n")
    print(f"  {'variants':>9}{'best Sharpe':>13}{'theory':>9}{'±1 se':>9}"
          f"{'deflated p':>12}{'verdict':>12}")
    print("  " + "-" * 65)

    prices = random_walk(N_BARS, seed=7)
    full = universe(max_slow=120)
    for n_trials in (1, 5, 25, 100, 500, 2000, len(full)):
        variants = full[:n_trials]
        params, best, _ = search(prices, variants)
        r = strategy_returns(prices, ma_crossover(prices, *params))
        se = sharpe_stderr(r)
        theory = expected_max_sharpe(n_trials, 1.0 / (len(r) - 1)) * math.sqrt(252)
        p = deflated_sharpe(r, n_trials)
        verdict = "SKILL" if p > 0.95 else "search"
        print(f"  {n_trials:>9,}{best:>13.3f}{theory:>9.3f}{se:>9.3f}"
              f"{p:>12.3f}{verdict:>12}")

    print("\n  Note the first row. At one variant there is no multiple testing to")
    print("  correct, so the test reduces to a Sharpe against zero and calls a")
    print("  0.605 over 2,000 bars significant — a false positive on pure noise.")
    print("  The correction addresses selection bias, not the ordinary error of a")
    print("  single estimate, and it is not advertised as doing both.")
    print("\n  Best-in-sample Sharpe climbs with the number of variants tried and")
    print("  tracks the theoretical maximum of that many null draws. Nothing was")
    print("  learned; the search space grew. Deflated Sharpe declines to call any")
    print("  of it skill, which on this data is the only correct answer.")

    # ---- power: what edge does it take to be believed? --------------------
    hdr("power: how large must a real edge be to survive the correction?")
    print("  A correction that rejects everything is not a correction, so the same")
    print("  500-variant search is run on markets with genuine AR(1) momentum of")
    print("  increasing strength. phi = 0 is the null and must be rejected.\n")
    print(f"  {'bars':>7}{'phi':>7}{'best Sharpe':>13}{'deflated p':>12}{'verdict':>10}")
    print("  " + "-" * 49)
    for n_bars in (2000, 5000, 20000):
        for phi in (0.00, 0.10, 0.20, 0.35):
            px = (with_momentum(n_bars, seed=7, phi=phi) if phi
                  else random_walk(n_bars, seed=7))
            variants = full[:500]
            params, best, _ = search(px, variants)
            r = strategy_returns(px, ma_crossover(px, *params))
            p = deflated_sharpe(r, len(variants))
            verdict = "SKILL" if p > 0.95 else "search"
            print(f"  {n_bars:>7,}{phi:>7.2f}{best:>13.3f}{p:>12.3f}{verdict:>10}")

    print("\n  It discriminates: phi = 0.35 is accepted at every sample size and the")
    print("  null is rejected at every sample size. But the bar is high. A phi of")
    print("  0.10 — a real, exploitable edge — is never accepted, not even with")
    print("  20,000 bars, and phi = 0.20 only reaches 0.925 there.")
    print("\n  That is the cost, stated plainly: trading false positives for false")
    print("  negatives. After a 500-variant search, a modest genuine edge is")
    print("  indistinguishable from having looked hard enough — and no amount of")
    print("  additional data fixes it, because searching 500 variants raises the")
    print("  bar by more than the extra data lowers it.")

    # ---- how often does a naive backtest 'discover' something? ------------
    hdr("false discovery rate of an uncorrected backtest")
    print("  200 independent random-walk markets. Each gets a 500-variant search.")
    print("  A naive backtest calls it a discovery when the best Sharpe beats 1.0.\n")

    naive_hits = deflated_hits = 0
    trials = 200
    for seed in range(trials):
        px = random_walk(N_BARS, seed=1000 + seed)
        variants = full[:500]
        params, best, _ = search(px, variants)
        r = strategy_returns(px, ma_crossover(px, *params))
        if best > 1.0:
            naive_hits += 1
        if deflated_sharpe(r, len(variants)) > 0.95:
            deflated_hits += 1

    print(f"  {'method':<34}{'discoveries':>13}{'rate':>9}")
    print("  " + "-" * 56)
    print(f"  {'naive: best Sharpe > 1.0':<34}{naive_hits:>13}"
          f"{naive_hits/trials:>8.1%}")
    print(f"  {'deflated Sharpe p > 0.95':<34}{deflated_hits:>13}"
          f"{deflated_hits/trials:>8.1%}")
    print(f"\n  True discoveries available in this data: 0. Every naive hit is a")
    print(f"  false positive produced entirely by searching.")


if __name__ == "__main__":
    main()
