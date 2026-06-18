import numpy as np

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Classification accuracy. Used only by LogisticRegression.
    Formula: number of correct predictions / total predictions.
    y_pred should be class labels (0 or 1), not probabilities.
    """
    return float(np.mean(y_true == y_pred))
