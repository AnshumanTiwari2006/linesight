import numpy as np
from linesight.utils.validators import _validate_X


def predict(self, X) -> np.ndarray:
    """Predict: y = dot(X_scaled, weights) + bias."""
    self._check_fitted("predict")
    X = _validate_X(X, expected_features=self._n_features_in_)
    if getattr(self, "_X_mean", None) is not None:
        X = (X - self._X_mean) / self._X_std
    return np.dot(X, self.weights) + self.bias
