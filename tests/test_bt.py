import math

import numpy as np
import pytest

from bt.cv import (KFold, PurgedKFold, forward_label_spans, leakage_fraction)
from bt.market import (ma_crossover, random_walk, returns, strategy_returns,
                       universe, with_momentum)
from bt.stats import (_norm_ppf, deflated_sharpe, expected_max_sharpe,
                      probabilistic_sharpe, sharpe, sharpe_stderr)


# ---------------------------------------------------------------------------
# the null: a market with no signal
# ---------------------------------------------------------------------------

def test_random_walk_returns_are_uncorrelated():
    """If the null had autocorrelation, every 'selection bias' result in this
    repo would partly be a real edge and the calibration would be void."""
    r = returns(random_walk(20_000, seed=3))
    ac = float(np.corrcoef(r[:-1], r[1:])[0, 1])
    assert abs(ac) < 0.03, ac


def test_momentum_generator_actually_has_momentum():
    """The other direction: the alternative hypothesis has to be real."""
    r = returns(with_momentum(20_000, seed=3, phi=0.2))
    ac = float(np.corrcoef(r[:-1], r[1:])[0, 1])
    assert ac > 0.1, ac


# ---------------------------------------------------------------------------
# no lookahead
# ---------------------------------------------------------------------------

def test_position_is_lagged_by_one_bar():
    """Trading the bar that generated the signal manufactures an edge from
    nothing and is the most common backtest bug there is."""
    p = random_walk(200, seed=1)
    pos = ma_crossover(p, 5, 20)
    assert pos[0] == 0.0
    # a position at t must be decidable from prices up to t-1
    p2 = p.copy()
    p2[100:] *= 1.5                       # change the future only
    assert np.array_equal(ma_crossover(p, 5, 20)[:100],
                          ma_crossover(p2, 5, 20)[:100])


def test_fast_must_be_shorter_than_slow():
    with pytest.raises(ValueError, match="shorter"):
        ma_crossover(random_walk(100), 20, 10)


def test_costs_reduce_returns():
    p = random_walk(500, seed=2)
    pos = ma_crossover(p, 5, 30)
    gross = strategy_returns(p, pos, cost_bps=0)
    net = strategy_returns(p, pos, cost_bps=10)
    assert net.sum() < gross.sum()


# ---------------------------------------------------------------------------
# Sharpe and its corrections
# ---------------------------------------------------------------------------

def test_sharpe_of_constant_returns_is_zero_not_infinite():
    assert sharpe(np.full(100, 0.01)) == 0.0


def test_sharpe_scales_with_annualisation():
    r = np.random.default_rng(0).normal(0.001, 0.01, 1000)
    assert sharpe(r, periods=252) == pytest.approx(sharpe(r, periods=1) * math.sqrt(252))


def test_norm_ppf_inverts_the_cdf():
    for p in (0.01, 0.1, 0.5, 0.9, 0.975, 0.999):
        x = _norm_ppf(p)
        back = 0.5 * (1 + math.erf(x / math.sqrt(2)))
        assert back == pytest.approx(p, abs=1e-6)


def test_expected_max_sharpe_matches_simulation():
    """The closed form is the load-bearing piece: it says how good the best of N
    worthless strategies looks. Checked against the actual maximum of N draws."""
    rng = np.random.default_rng(0)
    for n in (50, 200, 1000):
        sim = rng.standard_normal((3000, n)).max(axis=1).mean()
        assert expected_max_sharpe(n, 1.0) == pytest.approx(sim, abs=0.06)


def test_expected_max_sharpe_grows_without_bound():
    """Roughly sqrt(2 log N): it never stops rising, which is why a large enough
    search always finds something."""
    vals = [expected_max_sharpe(n, 1.0) for n in (10, 100, 1000, 10_000)]
    assert vals == sorted(vals)
    assert vals[-1] > vals[0] * 1.8


def test_expected_max_sharpe_is_zero_for_a_single_trial():
    assert expected_max_sharpe(1) == 0.0


def test_deflated_sharpe_falls_as_the_search_grows():
    """Same returns, more variants searched: the evidence must weaken."""
    r = np.random.default_rng(5).normal(0.0008, 0.01, 2000)
    ps = [deflated_sharpe(r, n) for n in (1, 10, 100, 1000, 10_000)]
    assert ps == sorted(ps, reverse=True)


def test_deflated_sharpe_rejects_a_search_over_pure_noise():
    """The headline property, across seeds rather than one.

    An earlier version asserted the best in-sample Sharpe exceeded 0.5 on a
    single seed. That number is a draw, not a property — seed 11 gives 0.394 —
    and pinning it would have been the same seed-fitting this repo exists to
    measure. What is a property: search inflates the winner above the field, and
    the correction rejects it anyway.
    """
    variants = universe(120)[:500]
    for seed in (11, 12, 13, 14):
        px = random_walk(2000, seed=seed)
        srs = [sharpe(strategy_returns(px, ma_crossover(px, *v))) for v in variants]
        best_i = int(np.argmax(srs))
        r = strategy_returns(px, ma_crossover(px, *variants[best_i]))

        # searching lifts the winner well clear of the typical variant
        assert srs[best_i] > float(np.median(srs)) + 0.3, seed
        # and it is still not skill
        assert deflated_sharpe(r, len(variants)) < 0.95, seed


def test_deflated_sharpe_accepts_a_large_genuine_edge():
    """A correction that rejects everything is not a correction."""
    px = with_momentum(2000, seed=7, phi=0.35)
    variants = universe(120)[:500]
    best = max(variants, key=lambda v: sharpe(strategy_returns(px, ma_crossover(px, *v))))
    r = strategy_returns(px, ma_crossover(px, *best))
    assert deflated_sharpe(r, len(variants)) > 0.95


def test_probabilistic_sharpe_is_deflated_sharpe_at_one_trial():
    r = np.random.default_rng(2).normal(0.0005, 0.01, 1000)
    assert deflated_sharpe(r, 1) == pytest.approx(probabilistic_sharpe(r), abs=1e-9)


def test_sharpe_stderr_shrinks_with_sample_size():
    rng = np.random.default_rng(1)
    errs = [sharpe_stderr(rng.normal(0.0005, 0.01, n)) for n in (250, 1000, 4000)]
    assert errs == sorted(errs, reverse=True)


# ---------------------------------------------------------------------------
# purged cross-validation
# ---------------------------------------------------------------------------

def test_plain_kfold_covers_every_index_exactly_once():
    seen = np.concatenate([te for _, te in KFold(5).split(100)])
    assert sorted(seen) == list(range(100))


def test_purging_removes_every_overlapping_training_row():
    """The defining property: no training label may share a bar with any test
    label."""
    n, h = 300, 20
    spans = forward_label_spans(n, h)
    for tr, te in PurgedKFold(5).split(n, spans):
        t_lo, t_hi = spans[te, 0].min(), spans[te, 1].max()
        assert not ((spans[tr, 0] <= t_hi) & (spans[tr, 1] >= t_lo)).any()


def test_embargo_removes_rows_after_the_test_block():
    n, h = 500, 5
    spans = forward_label_spans(n, h)
    plain = [len(tr) for tr, _ in PurgedKFold(5, 0.0).split(n, spans)]
    emb = [len(tr) for tr, _ in PurgedKFold(5, 0.10).split(n, spans)]
    assert sum(emb) < sum(plain)


def test_purging_costs_more_as_the_horizon_grows():
    n = 1000
    fracs = [leakage_fraction(n, forward_label_spans(n, h), 5)
             for h in (1, 10, 50, 200)]
    assert fracs == sorted(fracs)
    assert fracs[0] < 0.02 and fracs[-1] > 0.25


def test_horizon_one_needs_almost_no_purging():
    """With non-overlapping labels there is nothing to purge, so the correction
    must be nearly free — otherwise it is removing data for no reason."""
    n = 1000
    assert leakage_fraction(n, forward_label_spans(n, 1), 5) < 0.01


def test_purged_kfold_demands_spans():
    with pytest.raises(ValueError, match="spans"):
        list(PurgedKFold(5).split(100))


def test_rejects_bad_configuration():
    with pytest.raises(ValueError, match="at least 2"):
        KFold(1)
    with pytest.raises(ValueError, match="embargo"):
        PurgedKFold(5, embargo_pct=0.9)
