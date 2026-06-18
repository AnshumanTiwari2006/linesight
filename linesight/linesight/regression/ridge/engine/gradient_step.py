import numpy as np

def _gradient_step(self, X: np.ndarray, y: np.ndarray):
    """
    Ridge gradient step.
    Normal slope correction + L2 ridge penalty (bias never penalized).
    Gradient clipping prevents explosion on poorly scaled data.
    """
    n_samples = X.shape[0]
    y_pred = np.dot(X, self.weights) + self.bias
    residuals = y_pred - y

    # normal slope correction
    dw = (2 / n_samples) * np.dot(X.T, residuals)

    # ridge penalty shrinks weights (never the bias)
    ridge_penalty = 2 * self.alpha * self.weights
    dw += ridge_penalty

    db = (2 / n_samples) * np.sum(residuals)

    # Gradient clipping
    dw = np.clip(dw, -1e6, 1e6)
    db = float(np.clip(db, -1e6, 1e6))

    self.weights -= self.learning_rate * dw
    self.bias -= self.learning_rate * db
    return dw, db
