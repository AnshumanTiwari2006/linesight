import numpy as np

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error. Formula: (1/n) * sum((y_true - y_pred)^2)"""
    return float(np.mean((y_true - y_pred) ** 2))
