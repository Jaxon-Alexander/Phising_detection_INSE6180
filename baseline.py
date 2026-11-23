"""First milestone: a NumPy KNN baseline on development validation data."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import arff

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "Phishing_Legitimate_full.arff"
SHA256 = "052df52c43ce9900cc78f76d54e0e0296697b2c9c57f3ac0fcfa2990852467a9"


def load_data(path):
    """Load and verify the original benchmark.
    
    Args:
        path: Path to the unmodified Mendeley v1 ARFF file.
    
    Returns:
        A tuple of numeric features (10,000 x 48), binary labels, and
        the dataset SHA-256. Experiments treat class 1 as the positive class.
    
    Raises:
        ValueError: If the file hash, feature schema, or class counts differ."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SHA256:
        raise ValueError("Expected the original Mendeley v1 ARFF file (SHA-256 mismatch).")
    rows, meta = arff.loadarff(path)
    names = [name for name in meta.names() if name != "CLASS_LABEL"]
    x = np.column_stack([rows[name] for name in names])
    y = rows["CLASS_LABEL"].astype(int)
    if x.shape != (10000, 48) or not np.isfinite(x).all() or list(np.bincount(y)) != [5000, 5000]:
        raise ValueError("Unexpected benchmark schema or class counts.")
    return x, y, digest


def split_indices(x, y, seed=42):
    """Return reproducible training, validation, and test row indices.
    
    Identical full feature vectors stay in one partition, including vectors
    with conflicting labels. Groups are stratified by majority label, then
    allocated toward 64%, 16%, and 20% of the rows. The seed controls group order."""
    rng = np.random.default_rng(seed)
    _, group = np.unique(x, axis=0, return_inverse=True)
    group = group.ravel()
    size = np.bincount(group)
    positive = np.bincount(group, weights=y)
    # Stratify groups by majority class, treating tied groups separately.
    stratum = np.sign(2 * positive - size)
    split = np.empty(len(size), dtype=int)
    for value in np.unique(stratum):
        ids = rng.permutation(np.flatnonzero(stratum == value))
        midpoints = np.cumsum(size[ids]) - size[ids] / 2
        split[ids] = np.searchsorted([0.64, 0.80], midpoints / size[ids].sum(), side="right")
    return tuple(np.flatnonzero(split[group] == i) for i in range(3))


class KNN:
    """Binary K-nearest-neighbor classifier using squared Euclidean distance.
    
    Inputs have shape (n_samples, n_features). Callers must fit any scaling
    on training rows only. Scores are positive-label neighbor fractions,
    not calibrated probabilities. Odd k avoids ties in the class vote."""
    def __init__(self, k=5):
        """Configure the odd number of neighbors used for voting."""
        self.k = k

    def fit(self, x, y):
        """Store training features and labels after validating the neighbor count."""
        if self.k < 1 or self.k > len(x) or self.k % 2 == 0:
            raise ValueError("Choose a positive odd k no larger than the training set.")
        self.x, self.y = x, y
        return self

    def score_samples(self, queries):
        # Squared Euclidean distance; square roots do not change neighbor order.
        # Stable sorting breaks exact distance ties by training row order.
        """Return one positive-label vote fraction per query in bounded-size batches."""
        scores = []
        for batch in np.array_split(queries, max(1, (len(queries) + 63) // 64)):
            distances = (
                np.sum(batch ** 2, axis=1)[:, None]
                + np.sum(self.x ** 2, axis=1)
                - 2 * batch @ self.x.T
            )
            np.maximum(distances, 0, out=distances)
            closest = np.argsort(np.round(distances, 12), axis=1, kind="stable")[:, :self.k]
            scores.extend(self.y[closest].mean(axis=1))
        return np.array(scores)


def metrics(y, scores):
    """Compute binary classification metrics from positive-class scores.
    
    Scores at or above 0.5 predict class 1. The confusion matrix is
    [[true negatives, false positives], [false negatives, true positives]].
    ROC-AUC uses score ordering and gives half credit for tied scores.
    Undefined ratios return zero, including AUC for a single-class input."""
    pred = scores >= 0.5
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    tn = int(np.sum(~pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))

    def divide(a, b):
        return a / b if b else 0.0

    # Mann-Whitney form of ROC-AUC: ties count as half a win.
    _, bins = np.unique(scores, return_inverse=True)
    positives = np.bincount(bins, weights=y)
    negatives = np.bincount(bins, weights=1 - y)
    auc = divide(float(np.dot(positives, np.cumsum(negatives) - negatives / 2)),
                 (tp + fn) * (tn + fp))
    return dict(accuracy=divide(tp + tn, len(y)), precision=divide(tp, tp + fp),
                recall=divide(tp, tp + fn), f1=divide(2 * tp, 2 * tp + fp + fn),
                roc_auc=auc, confusion_matrix=[[tn, fp], [fn, tp]])


def main():
    """Run the original KNN validation milestone and save its split and metrics."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    x, y, digest = load_data(args.data)
    train, val, test = split_indices(x, y)
    # Estimate scaling parameters on development training rows only.
    # Validation and test statistics must not influence the fitted scaler.
    center = x[train].mean(axis=0)
    scale = x[train].std(axis=0)
    scale[scale == 0] = 1
    model = KNN(k=5).fit((x[train] - center) / scale, y[train])
    scores = model.score_samples((x[val] - center) / scale)
    result = dict(dataset_sha256=digest, seed=42, model="knn", k=5,
                  partition="development_validation", numpy_version=np.__version__,
                  split_counts={name: np.bincount(y[idx], minlength=2).tolist()
                                for name, idx in zip(("train", "validation", "test"),
                                                     (train, val, test))},
                  metrics=metrics(y[val], scores))
    args.output.mkdir(parents=True, exist_ok=True)
    # Keep row identities so every runner evaluates the same samples.
    np.savez_compressed(args.output / "splits.npz", train=train, validation=val, test=test)
    payload = json.dumps(result, indent=2)
    (args.output / "knn_validation.json").write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
