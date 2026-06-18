import numpy as np

def _compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
    y_pred = np.dot(X, self.weights) + self.bias
    mse_term = float(np.mean((y_pred - y) ** 2))
    l2_term = self.alpha * float(np.sum(self.weights ** 2))
    return mse_term + l2_term
