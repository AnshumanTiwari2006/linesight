import numpy as np


def _add_bias_column(X: np.ndarray) -> np.ndarray:
    """
    Prepend a column of ones to X for vectorized linear algebra.

    Turns X of shape (n, p) into shape (n, p+1) where column 0 is all ones.
    This lets the bias/intercept be treated as a regular coefficient:
        y_hat = X_bias @ theta
    where theta[0] is the intercept and theta[1:] are feature coefficients.

    Parameters
    ----------
    X : np.ndarray, shape (n, p)

    Returns
    -------
    np.ndarray, shape (n, p+1)
    """
    n = X.shape[0]
    ones = np.ones((n, 1))
    return np.hstack([ones, X])
