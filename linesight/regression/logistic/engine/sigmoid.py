import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Numerically stable sigmoid function.

    Formula: sigma(z) = 1 / (1 + exp(-z))

    Why numerical stability matters
    --------------------------------
    For very negative z (e.g. z = -800), exp(-z) = exp(800) overflows to inf,
    giving 1 / (1 + inf) = 0 which is actually correct, but numpy raises a
    RuntimeWarning and returns nan instead of 0.

    Solution: use two equivalent forms and switch based on sign of z:
      For z >= 0:  1 / (1 + exp(-z))         <- standard, no overflow
      For z < 0:   exp(z) / (1 + exp(z))     <- no overflow since exp(z) is small

    Parameters
    ----------
    z : np.ndarray of any shape — the linear combination X @ theta

    Returns
    -------
    np.ndarray of same shape — probabilities in range (0, 1)
    """
    return np.where(
        z >= 0,
        1.0 / (1.0 + np.exp(-z)),
        np.exp(z) / (1.0 + np.exp(z))
    )
