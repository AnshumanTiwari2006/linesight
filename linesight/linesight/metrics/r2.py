import numpy as np

def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    R-squared (coefficient of determination).

    Formula: 1 - SS_res / SS_tot
    where SS_res = sum((y_true - y_pred)^2)
    and   SS_tot = sum((y_true - mean(y_true))^2)

    Returns 1.0 for perfect fit.
    Returns 0.0 if model just predicts the mean.
    Can be negative if model is worse than predicting the mean.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        # All y values are identical — R^2 is undefined, return 0
        return 0.0
    return float(1 - ss_res / ss_tot)
