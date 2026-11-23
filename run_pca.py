"""Compare fixed baselines with PCA on the original validation split."""
import argparse
import json
from pathlib import Path

import numpy as np

from baseline import DATA, ROOT, load_data, metrics, split_indices
from run_baselines import baseline_models


class PCA:
    """Center, decompose the training covariance, and project without whitening."""

    def fit(self, x):
        """Estimate a training mean and descending covariance eigenvectors.
        
        x must be a finite numeric matrix with at least two rows. Scaling, when
        needed, belongs to the caller. No labels are used and no whitening is
        applied. Returns self, with axes and cumulative explained variance."""
        self.mean = x.mean(axis=0)
        centered = x - self.mean
        # The symmetric covariance matrix has real eigenvalues and orthogonal axes.
        values, vectors = np.linalg.eigh(centered.T @ centered / (len(x) - 1))
        values = np.maximum(values[::-1], 0)
        if values.sum() == 0:
            raise ValueError("PCA requires training features with nonzero variance.")
        vectors = vectors[:, ::-1]
        # Fix eigenvector signs to make the projection reproducible.
        pivots = np.argmax(np.abs(vectors), axis=0)
        self.axes = vectors * np.sign(vectors[pivots, np.arange(vectors.shape[1])])
        self.cumulative_variance = np.cumsum(values / values.sum())
        return self

    def transform(self, x, n_components):
        """Project rows onto the first n_components training-fitted axes."""
        return (x - self.mean) @ self.axes[:, :n_components]


def main():
    """Compare each baseline with PCA while preserving its original scaling policy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    x, y, digest = load_data(DATA)
    with np.load(ROOT / "results" / "splits.npz") as saved:
        indices = tuple(saved[key] for key in ("train", "validation", "test"))
    if any(not np.array_equal(a, b) for a, b in zip(indices, split_indices(x, y))):
        raise ValueError("Saved split does not match the original seed-42 split.")
    train, val, _ = indices
    raw_train, raw_val = x[train], x[val]
    mean, std = raw_train.mean(axis=0), raw_train.std(axis=0)
    std[std == 0] = 1
    y_train, y_val = y[train], y[val]
    # The variance target is fixed before looking at validation performance.
    target = 0.95
    out = dict(dataset_sha256=digest, partition="development_validation",
               split_seed=42, train_rows=len(train), validation_rows=len(val),
               original_features=x.shape[1], variance_target=target,
               numpy_version=np.__version__, models={})
    projections = {}
    for name, (cls, params, standardize) in baseline_models().items():
        a, b = ((raw_train - mean) / std, (raw_val - mean) / std) if standardize else (raw_train, raw_val)
        if standardize not in projections:
            pca = PCA().fit(a)
            k = int(np.searchsorted(pca.cumulative_variance, target)) + 1
            projections[standardize] = (pca.transform(a, k), pca.transform(b, k),
                                        k, float(pca.cumulative_variance[k - 1]))
        a_pca, b_pca, k, retained = projections[standardize]
        base = metrics(y_val, cls(**params).fit(a, y_train).score_samples(b))
        reduced = metrics(y_val, cls(**params).fit(a_pca, y_train).score_samples(b_pca))
        out["models"][name] = dict(parameters=params, standardized=standardize,
                                   components=k, variance_retained=retained,
                                   baseline=base, pca=reduced,
                                   f1_delta=reduced["f1"] - base["f1"],
                                   missed_phishing_delta=reduced["confusion_matrix"][1][0]
                                   - base["confusion_matrix"][1][0])
    args.output.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(out, indent=2)
    (args.output / "pca_validation.json").write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
