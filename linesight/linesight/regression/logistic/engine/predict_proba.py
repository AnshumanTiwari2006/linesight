import numpy as np
from linesight.utils.validators import _validate_X
from linesight.regression.logistic.engine.sigmoid import _sigmoid


def predict_proba(self, X) -> np.ndarray:
    """Return probability of class 1 for each sample."""
    self._check_fitted("predict_proba")
    X = _validate_X(X, expected_features=self._n_features_in_)
    if getattr(self, "_X_mean", None) is not None:
        X = (X - self._X_mean) / self._X_std
    return _sigmoid(np.dot(X, self.weights) + self.bias)
