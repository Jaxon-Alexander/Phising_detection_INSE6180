"""Compare training-only Pearson feature subsets with baseline and PCA."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import arff

from baseline import DATA, ROOT, load_data, metrics, split_indices
from run_baselines import baseline_models


def rank_features(x, y):
    """Return column ranks and absolute feature-label Pearson correlations.
    
    Use only training rows. Center each column and the binary target, then
    divide their absolute dot product by their norms. Constant columns
    receive zero. Stable sorting resolves equal scores by column index."""
    centered = x - x.mean(axis=0)
    labels = y - y.mean()
    denominator = np.linalg.norm(centered, axis=0) * np.linalg.norm(labels)
    scores = np.divide(np.abs(centered.T @ labels), denominator,
                       out=np.zeros(x.shape[1]), where=denominator > 0)
    return np.argsort(-scores, kind="stable"), scores


def main():
    """Evaluate nested training-ranked feature subsets and attach reference metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    x, y, digest = load_data(DATA)
    _, schema = arff.loadarff(DATA)
    names = [name for name in schema.names() if name != "CLASS_LABEL"]
    with np.load(ROOT / "results" / "splits.npz") as saved:
        indices = tuple(saved[key] for key in ("train", "validation", "test"))
    if any(not np.array_equal(a, b) for a, b in zip(indices, split_indices(x, y))):
        raise ValueError("Saved split does not match the original seed-42 split.")
    train, val, _ = indices
    a, b, labels = x[train], x[val], y[train]
    # Rank once on training labels; validation labels are used only for scoring.
    ranking, correlations = rank_features(a, labels)
    mean, scale = a.mean(axis=0), a.std(axis=0)
    scale[scale == 0] = 1
    scaled_a, scaled_b = (a - mean) / scale, (b - mean) / scale

    # Reuse verified earlier metrics as references, not as fitting data.
    refs = {}
    for name in ("baselines", "pca", "tree_pca"):
        report = json.loads((ROOT / "results" / f"{name}_validation.json").read_text())
        if (report["dataset_sha256"], report["split_seed"], report["partition"]) != (
                digest, 42, "development_validation"):
            raise ValueError(f"Reference experiment mismatch: {name}")
        refs[name] = report["models"]

    sizes = (12, 24, 34)
    output = dict(dataset_sha256=digest, split_seed=42, partition="development_validation",
                  train_rows=len(train), validation_rows=len(val), numpy_version=np.__version__,
                  ranking_method="absolute Pearson correlation on training data",
                  decision_threshold=0.5, subset_sizes=sizes,
                  feature_ranking=[dict(feature=names[i], abs_correlation=float(correlations[i]))
                                   for i in ranking], models={})
    for name, (model, params, standardize) in baseline_models().items():
        left, right = (scaled_a, scaled_b) if standardize else (a, b)
        base = refs["baselines"][name]
        if base["parameters"] != params or base["standardized"] != standardize:
            raise ValueError(f"Baseline configuration mismatch: {name}")
        base_metrics = base["metrics"]
        if name in refs["tree_pca"]:
            pca = refs["tree_pca"][name]["standardized_pca"]
            pca_dimensions, pca_metrics = pca["input_dimensions"], pca["metrics"]
        else:
            pca = refs["pca"][name]
            pca_dimensions, pca_metrics = pca["components"], pca["pca"]
        runs = {}
        for k in sizes:
            # Preserve original column order after selecting by rank.
            cols = np.sort(ranking[:k])
            scores = model(**params).fit(left[:, cols], labels).score_samples(right[:, cols])
            result = metrics(y[val], scores)
            runs[str(k)] = dict(metrics=result,
                                f1_delta_vs_baseline=result["f1"] - base_metrics["f1"],
                                missed_delta_vs_baseline=result["confusion_matrix"][1][0]
                                - base_metrics["confusion_matrix"][1][0])
        output["models"][name] = dict(parameters=params, standardized=standardize,
                                       baseline=base_metrics, pca_dimensions=pca_dimensions,
                                       standardized_pca=pca_metrics, subsets=runs)
        if name == "random_forest":
            output["models"][name]["candidates_per_split"] = {
                str(k): int(np.sqrt(k)) for k in (*sizes, x.shape[1])}
    args.output.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(output, indent=2)
    (args.output / "feature_selection_validation.json").write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
