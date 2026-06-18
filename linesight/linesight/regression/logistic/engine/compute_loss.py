import numpy as np
from linesight.regression.logistic.engine.sigmoid import _sigmoid

def _compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
    n_samples = X.shape[0]
    y_hat = _sigmoid(np.dot(X, self.weights) + self.bias)
    y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
    return float(-(1 / n_samples) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)))
