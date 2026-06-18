import numpy as np


def _expand_features(X: np.ndarray, degree: int) -> np.ndarray:
    """
    Expand a single-feature array into polynomial features.

    Input:  X shape (n, 1), degree=3
    Output: array shape (n, 3) with columns [x, x², x³]

    NOTE: The bias column (ones) is NOT added here.
    _add_bias_column() handles that in fit.py as usual.

    Parameters
    ----------
    X : np.ndarray, shape (n, 1) — single feature required
    degree : int — highest power to include (must be >= 1)

    Returns
    -------
    np.ndarray, shape (n, degree) — columns are x^1, x^2, ..., x^degree

    Why not include x^0 here
    ------------------------
    x^0 = 1 for all x, which is just the bias column. _add_bias_column()
    prepends that separately to keep the pattern consistent with all other
    regression types.
    """
    x = X.ravel()
    return np.column_stack([x ** p for p in range(1, degree + 1)])
