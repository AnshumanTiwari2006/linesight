import numpy as np
import warnings
from linesight.exceptions import LineSightShapeError, LineSightDataWarning


def _to_numpy(arr) -> np.ndarray:
    if hasattr(arr, "to_numpy"):
        return arr.to_numpy()
    if hasattr(arr, "values"):
        return arr.values
    return np.asarray(arr, dtype=float)


def _validate_X(X, expected_features: int = None) -> np.ndarray:
    X = _to_numpy(X)
    if X.ndim == 0:
        raise LineSightShapeError("X must be an array, not a scalar.")
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[0] == 0:
        raise LineSightShapeError("X has 0 samples.")
    if np.any(np.isnan(X)):
        raise LineSightShapeError(
            "X contains NaN values. Clean your data before training."
        )
    if np.any(np.isinf(X)):
        raise LineSightShapeError(
            "X contains Inf values. Clean your data before training."
        )
    if expected_features is not None and X.shape[1] != expected_features:
        raise LineSightShapeError(
            f"Model trained on {expected_features} feature(s), "
            f"but X has {X.shape[1]} feature(s)."
        )
    return X


def _validate_Xy(X, y) -> tuple:
    X = _validate_X(X)
    y = _to_numpy(y).ravel()
    if np.any(np.isnan(y)):
        raise LineSightShapeError("y contains NaN values.")
    if np.any(np.isinf(y)):
        raise LineSightShapeError("y contains Inf values.")
    if X.shape[0] != y.shape[0]:
        raise LineSightShapeError(
            f"X has {X.shape[0]} rows, y has {y.shape[0]} elements."
        )
    if X.shape[0] < 2:
        raise LineSightShapeError(
            "Training requires at least 2 samples. Got 1."
        )
    # Warn about zero-variance (constant) features
    zero_var = np.where(np.var(X, axis=0) == 0)[0]
    if len(zero_var) > 0:
        warnings.warn(
            f"Feature(s) at column index {list(zero_var)} have zero variance "
            f"(constant values). These carry no information.",
            LineSightDataWarning, stacklevel=3
        )
    return X, y


def _validate_binary_y(y: np.ndarray) -> None:
    """Strict binary check for logistic regression."""
    if not np.all(np.isin(y, [0, 1])):
        raise LineSightShapeError(
            f"Logistic regression requires y to contain only 0s and 1s. "
            f"Found unique values: {np.unique(y).tolist()}"
        )
