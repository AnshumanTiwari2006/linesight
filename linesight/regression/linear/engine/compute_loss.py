import numpy as np


def _compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Mean Squared Error for the current model state.

    Parameters
    ----------
    X : np.ndarray, shape (n, 1) — already validated, no bias column
    y : np.ndarray, shape (n,)

    Returns
    -------
    float — MSE value

    Internal note
    -------------
    Uses self.m and self.b directly.
    Does NOT add a bias column — linear regression stores intercept separately.
    """
    y_pred = self.m * X.ravel() + self.b
    return float(np.mean((y - y_pred) ** 2))
