import numpy as np

def _gradient_step(self, X: np.ndarray, y: np.ndarray):
    """
    Vectorized gradient step.
    dw = (2/n) * X.T @ residuals   (slope correction)
    db = (2/n) * sum(residuals)     (bias correction)
    Gradient clipping prevents explosion on poorly scaled data.
    """
    n_samples = X.shape[0]
    y_pred = np.dot(X, self.weights) + self.bias
    residuals = y_pred - y

    dw = (2 / n_samples) * np.dot(X.T, residuals)
    db = (2 / n_samples) * np.sum(residuals)

    # Gradient clipping: cap magnitude at 1e6 to prevent explosion
    dw = np.clip(dw, -1e6, 1e6)
    db = float(np.clip(db, -1e6, 1e6))

    self.weights -= self.learning_rate * dw
    self.bias -= self.learning_rate * db
    return dw, db
