import numpy as np
from linesight.utils.validators import _validate_X


def predict(self, X) -> np.ndarray:
    """
    Generate predictions for input X.

    Parameters
    ----------
    X : array-like, shape (n,) or (n, 1)

    Returns
    -------
    np.ndarray, shape (n,)
        Predicted y values using y_hat = coef_ * x + intercept_

    Raises
    ------
    LineSightNotFittedError if fit() has not been called.
    LineSightShapeError if X has wrong number of features.
    """
    self._check_fitted("predict")
    X = _validate_X(X, expected_features=self._n_features_in_)
    return self.m * X.ravel() + self.b
