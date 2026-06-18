import numpy as np

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error. Formula: (1/n) * sum(|y_true - y_pred|)"""
    return float(np.mean(np.abs(y_true - y_pred)))
