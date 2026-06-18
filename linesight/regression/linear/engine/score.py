import numpy as np
from linesight.utils.validators import _validate_Xy
from linesight.metrics import mse, rmse, mae, r2


def score(self, X, y) -> dict:
    """
    Compute all regression metrics and return as a dictionary.

    Parameters
    ----------
    X : array-like
    y : array-like

    Returns
    -------
    dict with keys:
        mse      : float — Mean Squared Error
        rmse     : float — Root Mean Squared Error
        mae      : float — Mean Absolute Error
        r2       : float — R-squared
        n_samples: int   — number of data points
        n_features: int  — number of input features

    All values rounded to 6 decimal places for readability.
    """
    self._check_fitted("score")
    X, y = _validate_Xy(X, y)
    y_pred = self.predict(X)

    return {
        "mse":       round(mse(y, y_pred), 6),
        "rmse":      round(rmse(y, y_pred), 6),
        "mae":       round(mae(y, y_pred), 6),
        "r2":        round(r2(y, y_pred), 6),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
    }
