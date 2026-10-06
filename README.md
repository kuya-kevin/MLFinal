# Model Selection for Driver-Mutation Classification

Predicting whether a mutant p53 protein is transcriptionally **active** (cancer-rescuing) or **inactive** (cancerous) from biophysical simulation features — where the class that matters is 0.9% of the data.

**Repository layout:** from-scratch and library implementations live in [`src/`](src/) — preprocessing ([`clean_data.py`](src/clean_data.py)), the three models ([`logistic_regression.py`](src/logistic_regression.py) · [`knn.py`](src/knn.py) · [`svm.py`](src/svm.py)), and the [`driver.py`](src/driver.py) that runs them end to end. Experiment notebooks are grouped under [`notebooks/`](notebooks/) by model: [`logistic-regression/`](notebooks/logistic-regression) (imbalanced, SMOTE, and SMOTE + undersampling variants), [`knn/`](notebooks/knn), [`svm/`](notebooks/svm), and [`exploration/`](notebooks/exploration) (PCA / t-SNE feature exploration).

*Authors: Minhang Xu, Kevin Elaba, Jiayi Li, Gweneth Ge.*

## Introduction

p53 is the most commonly mutated gene in human cancer. A mutation can leave the protein transcriptionally inactive — unable to do its tumor-suppressing job — but some mutations are "rescued" and stay functional. Being able to predict, from structure alone, which mutants stay active is a precision-oncology problem: it narrows an enormous space of possible mutations down to the ones worth validating in the lab.

We worked from the UCI *p53 Mutants* dataset: 16,772 mutant instances, each described by 5,408 biophysical features (2D electrostatic/surface descriptors plus 3D distance features) derived from simulation, with a binary label determined by in-vivo assay. After dropping rows with missing values, 16,592 instances remained.

The number that defined the entire project is the label distribution:

| Class | Meaning | Count | Share |
|---|---|---:|---:|
| 0 — inactive | cancerous, non-functional | 16,449 | 99.14% |
| 1 — active | transcriptionally competent (rescued) | 143 | **0.86%** |

A **115:1 imbalance**, and the rare class is exactly the one we care about. This inverts the usual intuition about a "good" model. A classifier that ignores the features entirely and labels everything inactive scores **99.1% accuracy** while finding zero rescued mutants — a perfect score on the metric and a useless model for the task. The real question of this project was never "how accurate can we get" but "how many of the 143 can we actually find, and at what cost in false alarms."

We implemented three classifiers — logistic regression (by hand and via scikit-learn), k-nearest neighbors, and a support vector machine — and treated the imbalance, not the algorithm, as the primary variable.

## Methods

### Preprocessing and dimensionality

[`clean_data.py`](clean_data.py) loads the raw `K8` matrix, maps `active`/`inactive` to `1`/`0`, drops an empty trailing column and any row containing a `?`, and writes `cleaned_K8.csv`. Everything downstream reads that file.

With 5,408 features and 16,592 rows, dimensionality was a concern, so we implemented PCA from scratch (mean-centering, covariance eigendecomposition, variance-ratio selection) in [`Logistic_regression.py`](Logistic_regression.py). **449 principal components capture 99% of the variance** — a 12x reduction in width. We also used a t-SNE projection (`feature_select.ipynb`) to look at the two classes in 2D; the minority class does not form a clean separable cluster, which previewed how hard recall was going to be.

### The three classifiers

- **Logistic regression** — implemented from scratch: sigmoid, batch gradient descent over 1,000 iterations at a 0.001 learning rate, and a 0.5 decision threshold. We ran the hand-rolled version head-to-head against scikit-learn's as a correctness check.
- **k-NN** — implemented from scratch (Euclidean distance, majority vote) and mirrored with scikit-learn's `KNeighborsClassifier` so we could sweep `k` from 3 to 29 cheaply.
- **SVM** — scikit-learn's `SVC` with an RBF kernel.

### Handling the imbalance

This is where most of the work went. We evaluated three regimes for each model:

1. **Untouched** — train on the raw 115:1 data.
2. **SMOTE** — synthesize minority-class examples until the training set is balanced.
3. **SMOTE + random undersampling** — a pipeline that oversamples the minority to a fraction, then undersamples the majority, which keeps the training set smaller and the decision boundary less dominated by synthetic points.

Resampling was applied to the **training split only** — the test set kept its natural 115:1 ratio, because inflating the test set with synthetic positives would have measured the wrong thing.

### Evaluation

Because accuracy is meaningless at this imbalance, we report **precision, recall, F1, and ROC-AUC** alongside it, with the confusion matrix as the ground truth. Recall (what fraction of true rescued mutants we catch) and ROC-AUC (ranking quality independent of threshold) are the metrics that actually track the goal.

## Results

The headline, and the whole point of the project, is in the first two rows: **the two highest-accuracy models are the two worst models.** Untreated SVM and k-NN clear 98.9% accuracy by essentially never predicting the positive class — ROC-AUC sits at the 0.50 coin-flip line. Every meaningful gain came from resampling, at the expected cost of precision.

| Model | Imbalance handling | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| *Predict-all-inactive (baseline)* | — | 0.991 | — | 0.00 | 0.00 | 0.50 |
| SVM | none | 0.989 | 0.00 | 0.00 | 0.00 | 0.50 |
| k-NN | none | 0.990 | 1.00 | 0.04 | 0.07 | 0.52 |
| Logistic regression | none | 0.993 | 0.79 | 0.33 | 0.46 | 0.66 |
| k-NN | SMOTE | 0.943 | 0.08 | 0.48 | 0.13 | 0.71 |
| **SVM** | **SMOTE + undersampling** | **0.991** | **0.62** | **0.48** | **0.54** | **0.74** |
| Logistic regression | SMOTE + undersampling | 0.985 | 0.34 | 0.70 | 0.46 | 0.84 |
| **Logistic regression** | **SMOTE** | **0.983** | **0.31** | **0.74** | **0.46** | **0.86** |

A few things the numbers make clear:

**Resampling is what moved the needle, not the choice of algorithm.** Untreated SVM predicts inactive for all 4,978 test cases: 0.00 recall, 0.00 F1, 0.50 AUC — numerically identical to the do-nothing baseline despite being a fitted model. The same SVM with SMOTE + undersampling jumps to 0.48 recall and the best F1 in the study (0.54). The algorithm didn't change; the training distribution did.

**Untreated k-NN produces the project's most seductive number: precision 1.00.** It is also its most misleading. k-NN was so conservative it made only a handful of positive predictions, got them right, and missed 96% of the rescued mutants (recall 0.04). Precision in isolation, like accuracy, rewards a model for refusing to do the hard part.

**Logistic regression with SMOTE gave the best ranking performance (ROC-AUC 0.86) and the best recall (0.74).** If the downstream use is "hand a biologist a ranked shortlist of candidate rescue mutants to assay," this is the model you want — it recovers nearly three-quarters of the true positives. The trade-off is precision of 0.31: most flagged candidates are false alarms, which is an acceptable price when a missed rescue mutant is far more costly than a wasted assay.

**There is a genuine precision/recall choice here, not a single best model.** SVM + resampling is the balanced operating point (precision 0.62, recall 0.48, top F1). LR + SMOTE is the high-recall point (recall 0.74, top AUC). Which one is "best" is a function of the lab's cost of a false negative versus a false positive, not of the ROC curve alone.

**Our from-scratch logistic regression held its own against scikit-learn.** On the SMOTE regime the hand-implemented model reached ROC-AUC 0.86 versus the library's 0.78 — a reassurance that the gradient-descent implementation was sound, and a reminder that two "logistic regressions" can differ meaningfully once regularization and solver defaults diverge.

## Conclusion

The most useful lesson from this project had nothing to do with p53 specifically. It was watching a metric lie. Three different models all posted ~99% accuracy, and in two of the three that number certified a classifier that found none of the cases the project existed to find. Once we stopped optimizing for accuracy and started reporting recall and ROC-AUC against a naturally-imbalanced test set, the ranking of the models completely reordered, and the modeling decision that mattered turned out to be the sampling strategy rather than the estimator.

If we took this further, the next steps would be cost-sensitive learning (class-weighted losses rather than resampling), threshold tuning driven by an explicit false-negative/false-positive cost, and feature selection on the 5,408 descriptors — the t-SNE view suggested the signal is diffuse, and a model that ranks features by discriminative power would be more interpretable to a biologist than 449 anonymous principal components.

As it stands: three classifiers, three imbalance regimes, one dataset where being right 99% of the time is the easiest way to build a worthless model.

---

### Data and reproduction

The dataset is not included in this repository. It is the [UCI *p53 Mutants* dataset](https://archive.ics.uci.edu/dataset/188/p53+mutants); download `K8.data` into the `src/` directory alongside the scripts.

```bash
cd src
python clean_data.py   # produces cleaned_K8.csv from K8.data
python driver.py       # runs logistic regression, k-NN, and SVM end to end
```

The notebooks reproduce each model's imbalance experiments and figures individually. Dependencies: `numpy`, `pandas`, `scikit-learn`, `imbalanced-learn`, `matplotlib`.

> If you use the dataset, cite Danziger et al. (2009), *Predicting Positive p53 Cancer Rescue Regions Using Most Informative Positive (MIP) Active Learning*, PLOS Computational Biology 5(9), e1000498.
