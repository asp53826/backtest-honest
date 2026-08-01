"""Cross-validation that does not hand the future to the training set.

Ordinary k-fold assumes observations are independent. Financial labels are not:
a label formed over the next `h` bars overlaps the next `h-1` observations, so a
training row adjacent to a test row shares most of its outcome. Split naively and
the model is scored on information it was trained on.

Two corrections, both from López de Prado:

* **purging** — drop training observations whose label interval overlaps any
  test label interval;
* **embargo** — additionally drop a short window *after* each test block, since
  serial correlation leaks in that direction even without overlap.

`bench/leakage.py` measures what each is worth on data containing no signal at
all, where any measured skill is definitionally leakage.
"""

from __future__ import annotations

import numpy as np


def forward_label_spans(n: int, horizon: int) -> np.ndarray:
    """Interval `[i, i+horizon]` for a label formed from the next `horizon` bars.

    Returned as an (n, 2) array of inclusive endpoints. The last `horizon`
    observations have no complete label and are clipped rather than dropped, so
    index arithmetic downstream stays simple; callers that need them excluded
    can slice.
    """
    starts = np.arange(n)
    ends = np.minimum(starts + horizon, n - 1)
    return np.column_stack([starts, ends])


class KFold:
    """Plain contiguous k-fold, deliberately wrong for this data.

    Present so the benchmark can measure what purging is worth rather than
    asserting it. Splits are contiguous rather than shuffled because shuffling a
    time series is a different and even larger mistake.
    """

    def __init__(self, n_splits: int = 5):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        self.n_splits = n_splits

    def split(self, n: int):
        bounds = np.linspace(0, n, self.n_splits + 1).astype(int)
        idx = np.arange(n)
        for i in range(self.n_splits):
            lo, hi = bounds[i], bounds[i + 1]
            test = idx[lo:hi]
            train = np.concatenate([idx[:lo], idx[hi:]])
            yield train, test


class PurgedKFold(KFold):
    """Contiguous k-fold with purging and an embargo.

    `spans[i] = (start, end)` is the interval the i-th label is formed over.
    A training observation is dropped when its interval intersects the interval
    spanned by the test block, which is the only condition under which the two
    can share an outcome.
    """

    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.0):
        super().__init__(n_splits)
        if not 0.0 <= embargo_pct < 0.5:
            raise ValueError("embargo_pct must be in [0, 0.5)")
        self.embargo_pct = embargo_pct

    def split(self, n: int, spans: np.ndarray | None = None):
        if spans is None:
            raise ValueError("PurgedKFold needs label spans; see forward_label_spans")
        if len(spans) != n:
            raise ValueError(f"spans has {len(spans)} rows for n={n}")

        embargo = int(n * self.embargo_pct)
        for train, test in super().split(n):
            t_lo = int(spans[test, 0].min())
            t_hi = int(spans[test, 1].max())

            starts, ends = spans[train, 0], spans[train, 1]
            # overlap iff the intervals intersect at all
            overlaps = (starts <= t_hi) & (ends >= t_lo)
            kept = train[~overlaps]

            if embargo:
                # serial correlation survives past the label window, so hold out
                # a further block immediately after the test set
                kept = kept[(kept <= t_hi) | (kept > t_hi + embargo)]

            yield kept, test


def leakage_fraction(n: int, spans: np.ndarray, n_splits: int = 5,
                     embargo_pct: float = 0.0) -> float:
    """Share of training rows that plain k-fold would have leaked.

    Useful on its own: it says how contaminated a naive split is *before* any
    model is fitted, and it scales with the label horizon exactly as the overlap
    argument predicts.
    """
    plain = KFold(n_splits)
    purged = PurgedKFold(n_splits, embargo_pct)
    total = dropped = 0
    for (tr_p, _), (tr_q, _) in zip(plain.split(n), purged.split(n, spans)):
        total += len(tr_p)
        dropped += len(tr_p) - len(tr_q)
    return dropped / total if total else 0.0
