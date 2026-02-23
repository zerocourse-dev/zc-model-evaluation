"""
Model Evaluation Dashboard

Implement the classes below. Each method has a description of what it should do.
Run `pytest` to check your solutions.

Suggested implementation order:
  1. ConfusionMatrix (foundation for understanding metrics)
  2. MetricsCalculator (standalone metric functions)
  3. CrossValidator (uses metrics to evaluate models across folds)
  4. LearningCurve (uses CrossValidator to study data size effects)
"""

import math
import random


class ConfusionMatrix:
    """Computes and stores a confusion matrix for binary classification.

    Positive class is 1, negative class is 0.

    Example:
        >>> cm = ConfusionMatrix([1, 0, 1, 0], [1, 0, 0, 0])
        >>> cm.true_positives()
        1
        >>> cm.accuracy()
        0.75
    """

    def __init__(self, y_true, y_pred):
        """Initialize with true and predicted labels.

        Computes TP, FP, TN, FN for binary classification where 1 is positive.

        @param y_true: list of true labels (0 or 1)
        @param y_pred: list of predicted labels (0 or 1)
        """
        raise NotImplementedError("Implement __init__")

    def true_positives(self):
        """Return the number of true positives.

        A true positive is when the true label is 1 and the predicted label is 1.

        @return: int
        """
        raise NotImplementedError("Implement true_positives")

    def false_positives(self):
        """Return the number of false positives.

        A false positive is when the true label is 0 and the predicted label is 1.

        @return: int
        """
        raise NotImplementedError("Implement false_positives")

    def true_negatives(self):
        """Return the number of true negatives.

        A true negative is when the true label is 0 and the predicted label is 0.

        @return: int
        """
        raise NotImplementedError("Implement true_negatives")

    def false_negatives(self):
        """Return the number of false negatives.

        A false negative is when the true label is 1 and the predicted label is 0.

        @return: int
        """
        raise NotImplementedError("Implement false_negatives")

    def matrix(self):
        """Return the confusion matrix as a 2x2 list of lists.

        Format: [[TN, FP], [FN, TP]]

        Example:
            >>> cm = ConfusionMatrix([1, 0, 1, 0], [1, 0, 0, 0])
            >>> cm.matrix()
            [[2, 0], [1, 1]]

        @return: list of lists (2x2)
        """
        raise NotImplementedError("Implement matrix")

    def accuracy(self):
        """Return the accuracy: (TP + TN) / total.

        Example:
            >>> cm = ConfusionMatrix([1, 0, 1, 0], [1, 0, 0, 0])
            >>> cm.accuracy()
            0.75

        @return: float
        """
        raise NotImplementedError("Implement accuracy")

    def precision(self):
        """Return the precision: TP / (TP + FP).

        Returns 0.0 if the denominator is 0.

        @return: float
        """
        raise NotImplementedError("Implement precision")

    def recall(self):
        """Return the recall: TP / (TP + FN).

        Returns 0.0 if the denominator is 0.

        @return: float
        """
        raise NotImplementedError("Implement recall")

    def f1_score(self):
        """Return the F1 score: 2 * precision * recall / (precision + recall).

        Returns 0.0 if the denominator is 0.

        Example:
            >>> cm = ConfusionMatrix([1, 1, 0, 0], [1, 0, 0, 0])
            >>> round(cm.f1_score(), 4)
            0.6667

        @return: float
        """
        raise NotImplementedError("Implement f1_score")


class CrossValidator:
    """K-fold cross-validation for evaluating model performance.

    Splits data into k folds, trains on k-1 folds, validates on the remaining fold.

    Example:
        >>> cv = CrossValidator(k=5, shuffle=True, random_seed=42)
        >>> results = cv.cross_validate(MyModel, {}, X, y, metric="accuracy")
        >>> results['mean']  # average accuracy across 5 folds
    """

    def __init__(self, k=5, shuffle=True, random_seed=42):
        """Initialize the cross-validator.

        @param k: int, number of folds (default 5)
        @param shuffle: bool, whether to shuffle indices before splitting (default True)
        @param random_seed: int, random seed for reproducibility (default 42)
        """
        raise NotImplementedError("Implement __init__")

    def split(self, X, y):
        """Generate train/validation splits for k-fold cross-validation.

        Yields (train_X, train_y, val_X, val_y) for each fold.
        If shuffle is True, shuffles indices using the random seed before splitting.

        Example:
            >>> cv = CrossValidator(k=3)
            >>> for train_X, train_y, val_X, val_y in cv.split(X, y):
            ...     print(len(train_X), len(val_X))

        @param X: list, feature data
        @param y: list, labels
        @yield: tuple of (train_X, train_y, val_X, val_y)
        """
        raise NotImplementedError("Implement split")

    def cross_validate(self, model_class, model_params, X, y, metric="accuracy"):
        """Run k-fold cross-validation on a model.

        For each fold:
          1. Create a fresh model: model_class(**model_params)
          2. Fit: model.fit(train_X, train_y)
          3. Predict: model.predict(val_X)
          4. Compute the metric using MetricsCalculator

        Supported metrics: "accuracy", "precision", "recall", "f1_score".

        Example:
            >>> cv = CrossValidator(k=5)
            >>> results = cv.cross_validate(DummyClassifier, {"strategy": "most_frequent"}, X, y)
            >>> len(results['scores'])
            5

        @param model_class: class with fit(X, y) and predict(X) methods
        @param model_params: dict, keyword arguments for model_class constructor
        @param X: list, feature data
        @param y: list, labels
        @param metric: str, metric name (default "accuracy")
        @return: dict with 'scores' (list of floats), 'mean' (float), 'std' (float)
        """
        raise NotImplementedError("Implement cross_validate")


class MetricsCalculator:
    """Collection of static methods for computing evaluation metrics.

    All methods are static and operate on lists of true and predicted values.

    Example:
        >>> MetricsCalculator.accuracy([1, 0, 1], [1, 0, 0])
        0.6666666666666666
    """

    @staticmethod
    def accuracy(y_true, y_pred):
        """Compute accuracy: proportion of correct predictions.

        Example:
            >>> MetricsCalculator.accuracy([1, 0, 1, 0], [1, 0, 0, 0])
            0.75

        @param y_true: list of true labels
        @param y_pred: list of predicted labels
        @return: float
        """
        raise NotImplementedError("Implement accuracy")

    @staticmethod
    def precision(y_true, y_pred):
        """Compute precision for binary classification (class 1 is positive).

        Precision = TP / (TP + FP). Returns 0.0 if denominator is 0.

        @param y_true: list of true labels (0 or 1)
        @param y_pred: list of predicted labels (0 or 1)
        @return: float
        """
        raise NotImplementedError("Implement precision")

    @staticmethod
    def recall(y_true, y_pred):
        """Compute recall for binary classification (class 1 is positive).

        Recall = TP / (TP + FN). Returns 0.0 if denominator is 0.

        @param y_true: list of true labels (0 or 1)
        @param y_pred: list of predicted labels (0 or 1)
        @return: float
        """
        raise NotImplementedError("Implement recall")

    @staticmethod
    def f1_score(y_true, y_pred):
        """Compute F1 score for binary classification.

        F1 = 2 * precision * recall / (precision + recall). Returns 0.0 if denominator is 0.

        @param y_true: list of true labels (0 or 1)
        @param y_pred: list of predicted labels (0 or 1)
        @return: float
        """
        raise NotImplementedError("Implement f1_score")

    @staticmethod
    def mean_squared_error(y_true, y_pred):
        """Compute mean squared error for regression.

        MSE = (1/n) * sum((y_true_i - y_pred_i)^2)

        Example:
            >>> MetricsCalculator.mean_squared_error([1, 2, 3], [1.1, 2.1, 2.9])
            0.010000000000000002

        @param y_true: list of true values
        @param y_pred: list of predicted values
        @return: float
        """
        raise NotImplementedError("Implement mean_squared_error")

    @staticmethod
    def r_squared(y_true, y_pred):
        """Compute R-squared (coefficient of determination) for regression.

        R^2 = 1 - (SS_res / SS_tot)
        where SS_res = sum((y_true_i - y_pred_i)^2)
        and SS_tot = sum((y_true_i - mean(y_true))^2)

        Returns 0.0 if SS_tot is 0.

        Example:
            >>> MetricsCalculator.r_squared([1, 2, 3], [1, 2, 3])
            1.0

        @param y_true: list of true values
        @param y_pred: list of predicted values
        @return: float
        """
        raise NotImplementedError("Implement r_squared")

    @staticmethod
    def roc_curve(y_true, y_scores):
        """Compute ROC curve points (TPR and FPR at various thresholds).

        Iterates thresholds from 0 to 1 in steps of 0.01 (plus min/max of scores).
        At each threshold, predictions are y_scores >= threshold.
        Computes TPR and FPR at each threshold.

        Example:
            >>> result = MetricsCalculator.roc_curve([1, 0, 1], [0.9, 0.1, 0.8])
            >>> sorted(result.keys())
            ['fpr', 'thresholds', 'tpr']

        @param y_true: list of true binary labels (0 or 1)
        @param y_scores: list of predicted scores/probabilities
        @return: dict with 'thresholds' (list), 'tpr' (list), 'fpr' (list)
        """
        raise NotImplementedError("Implement roc_curve")

    @staticmethod
    def auc(fpr, tpr):
        """Compute Area Under the Curve using the trapezoidal rule.

        Points should be sorted by fpr (ascending).

        Example:
            >>> MetricsCalculator.auc([0, 0, 1], [0, 1, 1])
            1.0

        @param fpr: list of false positive rates
        @param tpr: list of true positive rates
        @return: float
        """
        raise NotImplementedError("Implement auc")


class LearningCurve:
    """Computes learning curves to study the effect of training set size.

    Creates a model with increasing amounts of training data, evaluates
    using cross-validation, and reports train/validation scores.

    Example:
        >>> lc = LearningCurve(MyModel, {}, X, y)
        >>> results = lc.compute([10, 20, 50, 100])
        >>> results['train_scores']  # list of mean train scores per size
    """

    def __init__(self, model_class, model_params, X, y):
        """Initialize with a model factory and dataset.

        @param model_class: class with fit(X, y) and predict(X) methods
        @param model_params: dict, keyword arguments for model_class constructor
        @param X: list, feature data
        @param y: list, labels
        """
        raise NotImplementedError("Implement __init__")

    def compute(self, train_sizes, metric="accuracy", k=3):
        """Compute learning curves for given training set sizes.

        For each size in train_sizes:
          1. Take the first `size` samples from X and y
          2. Run k-fold cross-validation on that subset
          3. Also compute the training score (fit on train fold, predict on train fold)

        Example:
            >>> lc = LearningCurve(DummyClassifier, {"strategy": "most_frequent"}, X, y)
            >>> results = lc.compute([20, 40, 60])
            >>> len(results['train_sizes'])
            3

        @param train_sizes: list of ints, number of samples to use
        @param metric: str, metric name (default "accuracy")
        @param k: int, number of CV folds (default 3)
        @return: dict with 'train_sizes' (list), 'train_scores' (list), 'val_scores' (list)
        """
        raise NotImplementedError("Implement compute")
