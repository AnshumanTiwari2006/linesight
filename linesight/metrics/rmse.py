import numpy as np
from linesight.metrics.mse import mse

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error. Square root of MSE."""
    return float(np.sqrt(mse(y_true, y_pred)))
