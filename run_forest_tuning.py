"""Tune both feature widths with grouped CV inside the training split."""
import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np

from baseline import DATA, ROOT, load_data, metrics, split_indices
from run_baselines import baseline_models
from run_feature_selection import rank_features


def grouped_folds(x, y, n_folds=3, seed=42):
    """Return grouped cross-validation fit/evaluation row indices.
    
    Groups are identical full feature vectors. Within majority-label strata,
    process larger groups first and assign them to the least-loaded fold.
    A seeded shuffle breaks size ties. Class balance is approximate because
    groups cannot be divided."""
    _, groups = np.unique(x, axis=0, return_inverse=True)
    groups = groups.ravel()
    size = np.bincount(groups)
    positives = np.bincount(groups, weights=y)
    strata = np.sign(2 * positives - size)
    rng = np.random.default_rng(seed)
    assignment = np.empty(len(size), dtype=int)
    for stratum in np.unique(strata):
        counts = np.zeros(n_folds, dtype=int)
        candidates = rng.permutation(np.flatnonzero(strata == stratum))
        for group in candidates[np.argsort(-size[candidates], kind="stable")]:
            fold = int(np.argmin(counts))
            assignment[group] = fold
            counts[fold] += size[group]
    fold_ids = assignment[groups]
    return [(np.flatnonzero(fold_ids != i), np.flatnonzero(fold_ids == i))
            for i in range(n_folds)]


def selected_columns(x, y, width):
    """Select training-ranked features, returning indices in original column order."""
    if width == x.shape[1]:
        return np.arange(width)
    ranking, _ = rank_features(x, y)
    return np.sort(ranking[:width])


def main():
    """Select forest settings by training-only CV, then evaluate on validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    x, y, digest = load_data(DATA)
    with np.load(ROOT / "results" / "splits.npz") as saved:
        indices = tuple(saved[k] for k in ("train", "validation", "test"))
    if any(not np.array_equal(a, b) for a, b in zip(indices, split_indices(x, y))):
        raise ValueError("Saved split does not match the original seed-42 split.")
    train, val, _ = indices
    a, labels = x[train], y[train]
    folds = grouped_folds(a, labels)
    forest, baseline, _ = baseline_models()["random_forest"]
    # Both feature widths receive the same eight-configuration search budget.
    grid = [dict(n_trees=trees, max_depth=depth, min_leaf=leaf, seed=baseline["seed"])
            for trees, depth, leaf in product((25, 51), (12, 16), (2, 4))]
    result = dict(dataset_sha256=digest, split_seed=42, numpy_version=np.__version__,
                  partition="development_validation", train_rows=len(train),
                  validation_rows=len(val), cv_folds=len(folds), cv_seed=42,
                  fold_class_counts=[np.bincount(labels[b], minlength=2).tolist()
                                     for _, b in folds],
                  selection_rule="Highest mean CV phishing F1; grid order breaks ties",
                  threshold=0.5, models={})
    for width in (48, 34):
        fold_data = []
        for fit_rows, eval_rows in folds:
            # Refit ranking inside each fold to avoid supervised-selection leakage.
            cols = selected_columns(a[fit_rows], labels[fit_rows], width)
            fold_data.append((a[fit_rows][:, cols], labels[fit_rows],
                              a[eval_rows][:, cols], labels[eval_rows]))
        trials = []
        for params in grid:
            scores = [metrics(vy, forest(**params).fit(tx, ty).score_samples(vx))["f1"]
                      for tx, ty, vx, vy in fold_data]
            trials.append(dict(parameters=params, fold_f1=scores,
                               mean_f1=float(np.mean(scores))))
        # Only CV scores choose the winner; validation remains outside this search.
        winner = max(trials, key=lambda t: t["mean_f1"])
        cols = selected_columns(a, labels, width)

        def evaluate(params):
            scores = forest(**params).fit(a[:, cols], labels).score_samples(x[val][:, cols])
            return metrics(y[val], scores)

        base_metrics = evaluate(baseline)
        tuned = base_metrics if winner["parameters"] == baseline else evaluate(winner["parameters"])
        result["models"][str(width)] = dict(
            candidates_per_split=int(np.sqrt(width)), selected_columns=cols.tolist(),
            baseline_parameters=baseline, best_cv=winner, cv_trials=trials,
            baseline_validation=base_metrics, tuned_validation=tuned)
    args.output.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, indent=2)
    (args.output / "forest_tuning_validation.json").write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
