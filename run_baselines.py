"""Compare fixed classifier configurations on development validation only."""
import argparse
import json
from pathlib import Path

import numpy as np

from baseline import DATA, ROOT, KNN, load_data, metrics, split_indices
from models import DecisionTree, LogisticRegression, RandomForest


def baseline_models():
    """Return fresh model specifications shared by all experiment runners."""
    return {
        "knn": (KNN, {"k": 5}, True),
        "logistic_regression": (
            LogisticRegression, {"learning_rate": 0.05, "iterations": 1500, "l2": 0.001}, True),
        "decision_tree": (DecisionTree, {"max_depth": 16, "min_leaf": 2}, False),
        "random_forest": (
            RandomForest, {"n_trees": 51, "max_depth": 16, "min_leaf": 2, "seed": 42}, False),
    }


def main():
    """Evaluate fixed all-feature baselines on the saved development validation split."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    x, y, digest = load_data(DATA)
    expected = split_indices(x, y)
    # Reject changed partitions rather than silently comparing different samples.
    with np.load(ROOT / "results" / "splits.npz") as saved:
        train, val, test = (saved[key] for key in ("train", "validation", "test"))
    if any(not np.array_equal(a, b) for a, b in zip((train, val, test), expected)):
        raise ValueError("Saved split does not match the original seed-42 split.")
    x_train, x_val = x[train], x[val]
    center, scale = x_train.mean(axis=0), x_train.std(axis=0)
    scale[scale == 0] = 1
    results = dict(dataset_sha256=digest, partition="development_validation",
                   numpy_version=np.__version__, split_seed=42,
                   train_rows=len(train), validation_rows=len(val), test_rows=len(test),
                   feature_count=x.shape[1], decision_threshold=0.5, models={})
    for name, (cls, params, standardize) in baseline_models().items():
        a, b = ((x_train - center) / scale, (x_val - center) / scale) if standardize else (x_train, x_val)
        scores = cls(**params).fit(a, y[train]).score_samples(b)
        results["models"][name] = dict(parameters=params, standardized=standardize,
                                      metrics=metrics(y[val], scores))
    args.output.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(results, indent=2)
    (args.output / "baselines_validation.json").write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
