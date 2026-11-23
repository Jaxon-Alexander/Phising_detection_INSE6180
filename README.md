# Phishing website detection: baseline comparison

This starter verifies and loads the official 48-feature dataset, creates fixed
training, development validation, and final test partitions, and compares KNN, logistic regression, a decision tree, and Random Forest.
The classifiers use NumPy and Python. SciPy only reads ARFF.

## Run (Python 3.11 or newer)

Open a terminal in the extracted project directory:

    python -m venv .venv

Activate with `.\.venv\Scripts\Activate.ps1` on Windows PowerShell or
`source .venv/bin/activate` on Bash, then run:

    python -m pip install -r requirements.txt
    python run_baselines.py

The original ARFF is included in data/. To download it seperately, visit
https://data.mendeley.com/datasets/h3cgnj8hft/1 and place the original
Phishing_Legitimate_full.arff in data/. The script verifies its SHA-256.

## Benchmark protocol

- Source: Choon Lin Tan, *Phishing Dataset for Machine Learning: Feature Evaluation*,
  Mendeley Data, v1, 2018, DOI: 10.17632/h3cgnj8hft.1. Original ARFF, CC BY 4.0.
- Class 1 is phishing; class 0 is legitimate. Class mapping is also documented at
  https://ijci.journals.ekb.eg/article_392290_af4f40ca2484588b1827f82394730b7d.pdf
  (Table 1, dataset 2).
- Split uses seed 42 and groups identical feature vectors to avoid exact duplicate
  leakage. It retains all samples, including groups with conflicting labels.
- Train / validation / test: 6,400 / 1,600 / 2,000 rows. Keep the saved indices
  fixed. Development results are for debugging and method selection.
- Fit scaling only on the current training rows. Use all 48 features for this
  baseline. The KNN score is the fraction of phishing neighbors at k=5.
- The runner writes results/baselines_validation.json. To rerun only the original
  KNN milestone, use python baseline.py. Neither script scores the test set. The confusion matrix is [[TN, FP], [FN, TP]].

## Baselines

- KNN: Euclidean distance, 5 neighbors, training-only standardization.
- Logistic regression: 1,500 full-batch gradient steps, learning rate 0.05,
  L2 penalty 0.001, training-only standardization.
- Decision tree: weighted Gini splits, maximum depth 16, at least 2 rows per leaf.
- Random Forest: 51 bootstrap trees with the same depth and leaf settings,
  square root of feature count (rounded down) as the random candidate count
  at each split. The phishing score is the fraction of trees voting phishing.

These are fixed configurations for the baseline comparison. All use the same
48 features and a classification threshold of 0.5.

## Next experiments

1. Compare the all-feature baseline with PCA, feature selection alone, tuning
   alone, and feature selection plus tuning. Use the same classifiers across
   conditions and an explicit search budget. Use shared group-aware folds
   inside the training data. Fit scaling, PCA and selection inside each fold.
2. Freeze the configurations, refit on training plus validation, and evaluate
   once on the shared held-out test set. Report phishing F1 as the primary
   metric alongside accuracy, precision, recall, ROC-AUC and confusion matrices.

The paper methods are adapted to a controlled common protocol. Before choosing
exact PCA, selection or tuning settings, document the original papers' settings
and any differences from this experiment. Results measure performance on this
benchmark and split, with no claim of transfer to other datasets or to new sites.

## PCA-only experiment

Run `python run_pca.py` to reproduce `results/pca_validation.json`.
The runner refits each baseline and its PCA counterpart with the same
classifier settings on the same training rows, then scores the original
validation rows at threshold 0.5. No test predictions are made.

PCA uses the eigendecomposition of the training covariance matrix and retains
the smallest number of components explaining at least 95% of training
variance, without whitening. Both arms keep each model's original input
scaling: KNN and logistic regression use training-standardized inputs; the
tree models use raw inputs. This makes each pair a controlled comparison,
but the PCA representations differ across model families. Raw-input PCA
is sensitive to feature units, and retained variance does not measure how much
class information survived. The forest still uses the same square-root rule
for candidate features per split, so its candidate count falls from six to one
when the input drops from 48 features to three. These comparisons estimate the
effect of adding this PCA pipeline, not the isolated effect of rotation or
of any particular component count.

### Standardized PCA for tree models

Run `python run_tree_pca.py` to refit and compare the original tree models,
raw PCA, and PCA after training-only standardization. Results are written to
`results/tree_pca_validation.json`. The same 95% training variance target
retains 3 raw components and 34 standardized components. This compares two
complete preprocessing pipelines; it does not isolate scaling at a fixed
component count. The forest's unchanged square-root feature rule uses 6, 1,
and 5 candidates per split for the baseline, raw PCA, and standardized PCA,
respectively. The final test set is not scored.

### Supervised feature selection

Run `python run_feature_selection.py`. The runner computes absolute Pearson
correlations with the binary phishing label using only the original training
rows. It gives constant columns a score of zero, breaks ties by dataset column
order, and tests nested subsets of 12, 24, and 34 original features. Classifier
settings and the validation split are unchanged; the final test set is not scored.
The 34-feature subset can be compared with standardized PCA at equal input
width. The forest's square-root feature rule uses 3, 4, 5, and 6 split candidates
at widths 12, 24, 34, and 48, respectively.

`results/feature_selection_validation.json` records the full ranking, metrics,
confusion matrices, and baseline and PCA references. Correlation ranking scores
each feature individually, so it can keep redundant features or miss interactions.

### Random Forest tuning

Run `python run_forest_tuning.py`. It tests 8 configurations at each input
width (25 or 51 trees, depth 12 or 16, minimum leaf size 2 or 4). Both widths
use the same three folds inside the original training split. Identical
full-feature vectors stay in the same fold; for the 34-feature pipeline the
Pearson ranking is refit on each fold's training rows. The highest mean
phishing F1 wins, with ties broken by grid order. Each winner is refit on
the full training split and scored on the existing validation split, alongside
its untuned counterpart. No final test predictions are made.

`results/forest_tuning_validation.json` records the grid, fold scores, selected
columns, and validation metrics. The reduced forest considers five candidate
features per split, and the full forest considers six, under the unchanged
square-root rule. Report tuning as a comparison of these two complete
pipelines within this grid; the highest CV score does not guarantee an
improvement on validation or test.

### Final held-out comparison

Run `python run_final_test.py` from the extracted project directory. The
comparison is fixed at four classifiers × three input representations:
all 48 features, standardized PCA with a 95% variance target, and the top
34 original features by absolute Pearson correlation. The forest grid search
selected the original settings for both widths, so there is no separate
duplicate tuned arm.

The script pools the original training and validation rows (8,000 total),
then refits standardization, PCA, feature ranking, and all classifiers on
that pool. PCA component count and selected column identities are learned
again using the frozen rules. It scores the reserved 2,000 test rows and
writes `results/final_test.json` and `results/final_test.csv`. Rerunning the
same frozen script reproduces the comparison; do not use its test scores
to choose new settings. Keep development scores separate in the report.
