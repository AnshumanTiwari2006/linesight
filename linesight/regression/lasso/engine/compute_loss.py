import numpy as np

def _compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
    y_pred = np.dot(X, self.weights) + self.bias
    mse_term = float(np.mean((y - y_pred) ** 2))
    l1_term = self.alpha * float(np.sum(np.abs(self.weights)))
    return mse_term + l1_term
