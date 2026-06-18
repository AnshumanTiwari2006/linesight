import numpy as np


def _gradient_step(self, X: np.ndarray, y: np.ndarray) -> tuple:
    """
    Perform one gradient descent step. Updates self.m and self.b.

    Gradient derivation (MSE with respect to m and b):
    -------------------------------------------------------
    Loss J = (1/n) * sum((y_pred - y)^2)
           = (1/n) * sum((m*x + b - y)^2)

    dJ/dm = (2/n) * sum((m*x + b - y) * x)
    dJ/db = (2/n) * sum( m*x + b - y)

    Update rule:
        m := m - lr * dJ/dm
        b := b - lr * dJ/db

    Parameters
    ----------
    X : np.ndarray, shape (n, 1)
    y : np.ndarray, shape (n,)

    Returns
    -------
    tuple (grad_m, grad_b) — the raw gradient values BEFORE the update.
    Stored in history for analysis.
    """
    n = X.shape[0]
    x = X.ravel()
    y_pred = self.m * x + self.b

    residuals = y_pred - y

    grad_m = (2 / n) * np.sum(residuals * x)
    grad_b = (2 / n) * np.sum(residuals)

    self.m      -= self.learning_rate * grad_m
    self.b -= self.learning_rate * grad_b

    return (grad_m, grad_b)
