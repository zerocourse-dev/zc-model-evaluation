# Model Evaluation Dashboard

A ZeroCourse project for Course 14.1: Supervised Learning (Week 2).

## What You'll Build

Implement a complete model evaluation toolkit from scratch -- no sklearn allowed for the evaluation logic. You'll build four classes that work together to evaluate classifier performance:

### ConfusionMatrix

| Method | Description |
|--------|-------------|
| `__init__(y_true, y_pred)` | Store labels and compute TP, FP, TN, FN |
| `true_positives()` | Count of true positives |
| `false_positives()` | Count of false positives |
| `true_negatives()` | Count of true negatives |
| `false_negatives()` | Count of false negatives |
| `matrix()` | 2x2 confusion matrix as `[[TN, FP], [FN, TP]]` |
| `accuracy()` | (TP + TN) / total |
| `precision()` | TP / (TP + FP), 0 if undefined |
| `recall()` | TP / (TP + FN), 0 if undefined |
| `f1_score()` | Harmonic mean of precision and recall |

### MetricsCalculator

| Method | Description |
|--------|-------------|
| `accuracy(y_true, y_pred)` | Proportion of correct predictions |
| `precision(y_true, y_pred)` | Binary precision (class 1 = positive) |
| `recall(y_true, y_pred)` | Binary recall |
| `f1_score(y_true, y_pred)` | Binary F1 score |
| `mean_squared_error(y_true, y_pred)` | MSE for regression |
| `r_squared(y_true, y_pred)` | Coefficient of determination |
| `roc_curve(y_true, y_scores)` | TPR/FPR at thresholds 0..1 step 0.01 |
| `auc(fpr, tpr)` | Area under the curve (trapezoidal rule) |

### CrossValidator

| Method | Description |
|--------|-------------|
| `__init__(k, shuffle, random_seed)` | Configure fold count, shuffling, seed |
| `split(X, y)` | Yield `(train_X, train_y, val_X, val_y)` for each fold |
| `cross_validate(model_class, model_params, X, y, metric)` | Run k-fold CV, return `{scores, mean, std}` |

### LearningCurve

| Method | Description |
|--------|-------------|
| `__init__(model_class, model_params, X, y)` | Store model factory and dataset |
| `compute(train_sizes, metric, k)` | Return `{train_sizes, train_scores, val_scores}` |

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the tests (they will all fail initially):
   ```bash
   pytest tests/ --tb=short
   ```

3. Open `lib/model_evaluation.py` and implement each class.

4. Run the tests again to check your progress.

## Suggested Implementation Order

1. **ConfusionMatrix** -- Start here. Count TP/FP/TN/FN in `__init__`, then the metric methods are just arithmetic on those counts.

2. **MetricsCalculator** -- Build on the same ideas as ConfusionMatrix but as standalone static methods. Start with `accuracy`, then `precision`/`recall`/`f1_score`. Do `mean_squared_error` and `r_squared` next (simple formulas). Save `roc_curve` and `auc` for last in this class.

3. **CrossValidator** -- `split` is the core: shuffle indices, divide into k chunks, yield train/val pairs. `cross_validate` just calls `split`, fits a model each fold, and uses MetricsCalculator.

4. **LearningCurve** -- Uses CrossValidator internally. For each training size, slice the data and run CV.

## Tips

- **Binary classification convention**: Class 1 is positive, class 0 is negative. All metrics use this convention.
- **Zero division**: Whenever a denominator is 0 (e.g., no positive predictions), return 0.0 instead of raising an error.
- **ROC curve**: At each threshold `t`, classify as positive if `score >= t`. Sweep from 0 to 1 in steps of 0.01.
- **AUC via trapezoidal rule**: `sum of (x[i+1] - x[i]) * (y[i+1] + y[i]) / 2` over sorted FPR/TPR pairs.
- **Cross-validation shuffle**: Use `random.Random(seed)` for reproducibility. Create a shuffled list of indices, then split into consecutive chunks.
- **Standard deviation**: Use population std (divide by N, not N-1).
- Models passed to CrossValidator have a `fit(X, y)` and `predict(X)` interface -- just call them.

## Running Tests

```bash
pytest tests/                                # Run all tests
pytest tests/ -v                             # Verbose output
pytest tests/ -k "TestConfusionMatrix"       # Run one test class
pytest tests/ -k "test_accuracy"             # Run tests matching a name
```
