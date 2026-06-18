import numpy as np
from linesight.utils.validators import _validate_X
from linesight.utils.array_utils import _add_bias_column
from linesight.regression.polynomial.engine.expand_features import _expand_features


def predict(self, X) -> np.ndarray:
    self._check_fitted("predict")
    X = _validate_X(X, expected_features=1)
    X_poly = _expand_features(X, self.degree)
    if getattr(self, "normalize", False) and getattr(self, "_X_mean", None) is not None:
        X_poly = (X_poly - self._X_mean) / self._X_std
    X_bias = _add_bias_column(X_poly)
    return X_bias @ self.theta_
