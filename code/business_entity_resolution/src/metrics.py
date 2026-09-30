"""Evaluation metrics.

`f05_one` and `macro_f05` implement the challenge metric exactly (macro-averaged
F0.5 per Source 1 record, singletons included) and are the only objective used for
tuning. The other helpers are diagnostics reported alongside it.
"""


def f05_one(T: set, P: set) -> float:
    """F0.5 of one S1 record given its true set T and predicted set P."""
    if not T and not P: return 1.0          # correct "no match"
    if not T or not P:  return 0.0          # predicted match on singleton, or missed all
    tp = len(T & P)
    if tp == 0: return 0.0
    p, r = tp / len(P), tp / len(T)
    return 1.25 * p * r / (0.25 * p + r)


def macro_f05(truth: dict, pred: dict) -> float:
    """Mean F0.5 over every S1 key in `truth`; S1 records missing from `pred` count as empty."""
    return sum(f05_one(set(T), set(pred.get(k, []))) for k, T in truth.items()) / len(truth)


def singleton_accuracy(truth: dict, pred: dict) -> float:
    """Share of true singletons (empty truth) for which nothing was predicted."""
    singles = [k for k, T in truth.items() if not T]
    if not singles:
        return float("nan")
    return sum(not pred.get(k) for k in singles) / len(singles)


def pair_precision_recall(truth: dict, pred: dict) -> tuple[float, float]:
    """Micro (pair-level) precision and recall over all predicted / true pairs."""
    tp = n_pred = n_true = 0
    for k, T in truth.items():
        T, P = set(T), set(pred.get(k, []))
        tp += len(T & P)
        n_pred += len(P)
        n_true += len(T)
    return (tp / n_pred if n_pred else float("nan"),
            tp / n_true if n_true else float("nan"))


if __name__ == "__main__":
    # Worked example from the problem statement: expected F0.5 = 0.714.
    score = f05_one({"S2-00047", "S3-00812"}, {"S2-00047", "S2-00193", "S3-00812"})
    assert abs(score - 0.714) < 1e-3, score
    assert f05_one(set(), set()) == 1.0 and f05_one(set(), {"S2-1"}) == 0.0
    print(f"metrics self-test passed (example F0.5 = {score:.3f})")
