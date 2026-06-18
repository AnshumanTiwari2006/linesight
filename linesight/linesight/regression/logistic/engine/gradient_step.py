import numpy as np
from linesight.regression.logistic.engine.sigmoid import _sigmoid

def _gradient_step(self, X: np.ndarray, y: np.ndarray):
    """
    Logistic gradient step using binary cross-entropy.
    dw = (1/n) * X.T @ (y_hat - y)
    db = (1/n) * sum(y_hat - y)
    """
    n_samples = X.shape[0]
    y_hat = _sigmoid(np.dot(X, self.weights) + self.bias)

    dw = (1 / n_samples) * np.dot(X.T, (y_hat - y))
    db = (1 / n_samples) * np.sum(y_hat - y)

    dw = np.clip(dw, -1e6, 1e6)
    db = float(np.clip(db, -1e6, 1e6))

    self.weights -= self.learning_rate * dw
    self.bias -= self.learning_rate * db
    return dw, db
