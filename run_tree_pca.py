"""Compare tree baselines with raw and standardized PCA on validation only."""
import argparse
import json
from pathlib import Path

import numpy as np

from baseline import DATA, ROOT, load_data, metrics, split_indices
from run_baselines import baseline_models
from run_pca import PCA


def main():
    """Compare original tree inputs, raw PCA, and training-standardized PCA."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    x, y, digest = load_data(DATA)
    with np.load(ROOT / "results" / "splits.npz") as saved:
        indices = tuple(saved[key] for key in ("train", "validation", "test"))
    if any(not np.array_equal(a, b) for a, b in zip(indices, split_indices(x, y))):
        raise ValueError("Saved split does not match the original seed-42 split.")
    train, val, _ = indices
    a, b = x[train], x[val]
    mean, std = a.mean(axis=0), a.std(axis=0)
    std[std == 0] = 1
    # Equal variance targets may retain different widths under different scaling.
    target = 0.95
    arms = {"baseline": (a, b, None)}
    for name, left, right in (
        ("raw_pca", a, b),
        ("standardized_pca", (a - mean) / std, (b - mean) / std),
    ):
        pca = PCA().fit(left)
        k = int(np.searchsorted(pca.cumulative_variance, target)) + 1
        arms[name] = (pca.transform(left, k), pca.transform(right, k),
                      float(pca.cumulative_variance[k - 1]))
    result = dict(dataset_sha256=digest, partition="development_validation",
                  split_seed=42, train_rows=len(train), validation_rows=len(val),
                  variance_target=target, numpy_version=np.__version__, models={})
    for model in ("decision_tree", "random_forest"):
        cls, params, _ = baseline_models()[model]
        runs = {}
        for name, (left, right, retained) in arms.items():
            scores = cls(**params).fit(left, y[train]).score_samples(right)
            runs[name] = dict(input_dimensions=left.shape[1], variance_retained=retained,
                              metrics=metrics(y[val], scores))
        result["models"][model] = dict(parameters=params, **runs)
    args.output.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, indent=2)
    (args.output / "tree_pca_validation.json").write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
