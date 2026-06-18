import numpy as np

def _compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
    y_pred = np.dot(X, self.weights) + self.bias
    mse_term = float(np.mean((y - y_pred) ** 2))
    l1_term = self.alpha * self.l1_ratio * float(np.sum(np.abs(self.weights)))
    l2_term = self.alpha * (1 - self.l1_ratio) * float(np.sum(self.weights ** 2))
    return mse_term + l1_term + l2_term
