"""Refit the frozen comparison on development data and score the held-out test."""
import csv
import json

import numpy as np

from baseline import DATA, ROOT, load_data, metrics, split_indices
from run_baselines import baseline_models
from run_feature_selection import rank_features
from run_pca import PCA


def main():
    """Refit the frozen 12 pipelines and export held-out test metrics.
    
    The original training and validation partitions form the development
    pool. Scaling, PCA, and ranking are learned again on that pool only.
    The test split is scored after fitting, and never selects settings.
    Re-execution writes the same output paths; it is not a new tuning round."""
    output = ROOT / "results"
    x, y, digest = load_data(DATA)
    with np.load(output / "splits.npz") as saved:
        indices = tuple(saved[key] for key in ("train", "validation", "test"))
    if any(not np.array_equal(a, b) for a, b in zip(indices, split_indices(x, y))):
        raise ValueError("Saved split does not match the original seed-42 split.")
    train, val, test = indices
    # Configuration selection is finished; pool development rows for the final fit.
    dev = np.sort(np.concatenate((train, val)))
    a, b = x[dev], x[test]
    mean, std = a.mean(axis=0), a.std(axis=0)
    std[std == 0] = 1
    za, zb = (a - mean) / std, (b - mean) / std
    # Relearn column identities using the frozen top-34 selection rule.
    ranking, _ = rank_features(a, y[dev])
    cols = np.sort(ranking[:34])
    pca = PCA().fit(za)
    target = 0.95
    # The frozen variance target determines width from development data alone.
    count = int(np.searchsorted(pca.cumulative_variance, target)) + 1
    projection = pca.transform(za, count), pca.transform(zb, count)
    result = dict(dataset_sha256=digest, split_seed=42, partition="held_out_test",
                  fit_rows=len(dev), test_rows=len(test), decision_threshold=0.5,
                  pca_variance_target=target,
                  pca_variance_retained=float(pca.cumulative_variance[count - 1]),
                  pearson_columns=cols.tolist(), numpy_version=np.__version__, runs=[])
    for name, (cls, params, standardize) in baseline_models().items():
        left, right = (za, zb) if standardize else (a, b)
        arms = {
            "all_features": (left, right, standardize),
            "standardized_pca": (*projection, True),
            "pearson_top_34": (left[:, cols], right[:, cols], standardize),
        }
        # Every arm sees the same held-out rows and the same decision threshold.
        for arm, (fit_x, test_x, scaled) in arms.items():
            scores = cls(**params).fit(fit_x, y[dev]).score_samples(test_x)
            run = dict(model=name, input=arm, dimensions=fit_x.shape[1],
                       standardized=scaled, parameters=params, metrics=metrics(y[test], scores))
            if name == "random_forest":
                run["candidates_per_split"] = int(np.sqrt(fit_x.shape[1]))
            result["runs"].append(run)
    payload = json.dumps(result, indent=2)
    (output / "final_test.json").write_text(payload + "\n", encoding="utf-8")
    with (output / "final_test.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        keys = ("accuracy", "precision", "recall", "f1", "roc_auc")
        writer.writerow(("model", "input", "dimensions", *keys, "tn", "fp", "fn", "tp"))
        for run in result["runs"]:
            m = run["metrics"]
            (tn, fp), (fn, tp) = m["confusion_matrix"]
            writer.writerow((run["model"], run["input"], run["dimensions"],
                             *(m[key] for key in keys), tn, fp, fn, tp))
    print(payload)


if __name__ == "__main__":
    main()
