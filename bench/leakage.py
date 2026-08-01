"""What overlapping labels are worth to a model that should know nothing.

    uv run python bench/leakage.py

Features are built only from past draws and labels only from future draws, so
no genuine relationship exists. A correctly validated model must score zero, and
anything above zero is the future arriving through the training set.
"""

from __future__ import annotations

import numpy as np

from bt.cv import KFold, PurgedKFold, forward_label_spans, leakage_fraction


def hdr(t):
    print(f"\n=== {t} ===\n")


def make_problem(n: int, horizon: int, seed: int, smooth: int = 20):
    """Features from the past, labels from the future — independent by
    construction, and still leakable.

    Two properties are needed and the first version had only one:

    * labels **overlap** — y_t and y_{t+1} are built from the same future draws;
    * features are **autocorrelated**, so observations close in time are close
      in feature space.

    Without the second, a nearest-neighbour model picks temporally random
    training rows, never lands on one whose label overlaps the test label, and
    plain k-fold scores zero — which is what the first version of this benchmark
    measured, and it measured nothing. Real financial features are smoothed
    prices and moving averages; temporal proximity *is* feature proximity, and
    that is precisely what makes overlapping labels exploitable.

    No genuine edge exists: X_t is built only from draws before t and y_t only
    from draws at or after t.
    """
    rng = np.random.default_rng(seed)
    past = rng.normal(size=n + smooth)
    future = rng.normal(size=n + horizon)

    y = np.array([future[i:i + horizon].sum() for i in range(n)])

    # smoothed past noise: autocorrelated, and independent of `future`
    base = np.array([past[i:i + smooth].mean() for i in range(n)])
    X = np.column_stack([
        base,
        np.roll(base, 3),
        np.cumsum(base) / (np.arange(n) + 1),
        base ** 2,
    ])
    return X, y, forward_label_spans(n, horizon)


def knn_score(X_tr, y_tr, X_te, y_te, k: int = 5) -> float:
    """Correlation between prediction and truth, from a 1-NN-style memoriser.

    A high-capacity model is used deliberately: leakage is only visible through
    a model able to exploit it, and a linear fit on independent features cannot.
    """
    if len(X_tr) < k or len(X_te) == 0:
        return 0.0
    d = ((X_te[:, None, :] - X_tr[None, :, :]) ** 2).sum(-1)
    nn = np.argsort(d, axis=1)[:, :k]
    pred = y_tr[nn].mean(axis=1)
    if pred.std() < 1e-12 or y_te.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(pred, y_te)[0, 1])


def evaluate(splitter, n, X, y, spans=None) -> float:
    scores = []
    gen = splitter.split(n, spans) if spans is not None else splitter.split(n)
    for tr, te in gen:
        if len(tr) == 0:
            continue
        scores.append(knn_score(X[tr], y[tr], X[te], y[te]))
    return float(np.mean(scores)) if scores else 0.0


def main():
    N = 1200
    SEEDS = range(60)

    hdr("features carry no information; any score is leakage")
    print("  Labels are forward sums over `horizon` bars, so adjacent labels share")
    print("  most of their outcome. Features come only from past draws. The")
    print("  correct answer at every horizon is a score of zero.\n")
    print(f"  {'horizon':>8}{'plain k-fold':>18}{'purged':>18}"
          f"{'purged+embargo':>20}{'dropped':>10}")
    print("  " + "-" * 76)

    def ms(v):
        a = np.asarray(v)
        return a.mean(), a.std(ddof=1) / np.sqrt(len(a))

    for horizon in (1, 5, 20, 50, 100):
        plain, purged, emb = [], [], []
        for seed in SEEDS:
            X, y, spans = make_problem(N, horizon, seed)
            plain.append(evaluate(KFold(5), N, X, y))
            purged.append(evaluate(PurgedKFold(5, 0.0), N, X, y, spans))
            emb.append(evaluate(PurgedKFold(5, 0.02), N, X, y, spans))
        frac = leakage_fraction(N, forward_label_spans(N, horizon), 5, 0.02)
        pm, ps = ms(plain); qm, qs = ms(purged); em, es = ms(emb)
        print(f"  {horizon:>8}{pm:>12.4f} ±{ps:<5.4f}{qm:>12.4f} ±{qs:<5.4f}"
              f"{em:>14.4f} ±{es:<5.4f}{frac:>9.1%}")

    print(f"\n  {len(list(SEEDS))} seeds; ± is the standard error of the mean.")
    print("\n  Plain k-fold reports skill that cannot exist, and it grows with the")
    print("  label horizon exactly as the overlap argument predicts — nothing at")
    print("  horizon 1 where labels do not overlap, and clearly non-zero by 100.")
    print("\n  Purging removes most of it but the residual is not reliably zero.")
    print("  Read the error bars before calling that leftover real: the correction")
    print("  is worth what the gap between the columns says it is worth, and no")
    print("  more than the standard errors will support.")

    hdr("the cost side: purging is not free")
    print("  Dropped training rows against label horizon, five folds.\n")
    print(f"  {'horizon':>8}{'purged only':>13}{'+2% embargo':>13}"
          f"{'+5% embargo':>13}")
    print("  " + "-" * 47)
    for horizon in (1, 5, 20, 50, 100, 200):
        spans = forward_label_spans(N, horizon)
        a = leakage_fraction(N, spans, 5, 0.0)
        b = leakage_fraction(N, spans, 5, 0.02)
        c = leakage_fraction(N, spans, 5, 0.05)
        print(f"  {horizon:>8}{a:>12.1%}{b:>13.1%}{c:>13.1%}")

    print("\n  At a 200-bar horizon on 1,200 bars, purging removes most of the")
    print("  training set — which is the honest answer rather than a failure. A")
    print("  label reaching that far forward genuinely does contaminate almost")
    print("  everything adjacent to it, and a backtest that keeps those rows is")
    print("  not training on more data, it is training on the test set.")


if __name__ == "__main__":
    main()
