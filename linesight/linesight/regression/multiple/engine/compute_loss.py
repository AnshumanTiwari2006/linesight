import numpy as np

def _compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
    y_pred = np.dot(X, self.weights) + self.bias
    return float(np.mean((y_pred - y) ** 2))
