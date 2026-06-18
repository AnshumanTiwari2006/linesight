from linesight.metrics import mse, rmse, mae, r2
from linesight.utils.validators import _validate_Xy
import numpy as np


def explain_fit(self, X, y) -> str:
    """
    Print a readable summary of model fit quality.

    Interprets each metric in plain English so the reader understands
    what the numbers mean, not just what they are.

    Output example:
    ---------------
    Model fit summary
    -----------------
    R^2   = 0.9980
         The model explains 99.8% of the variance in y.
         (1.0 = perfect fit, 0.0 = no better than predicting the mean)

    RMSE = 0.3500
         On average, predictions are off by +/-0.35 in the same units as y.

    MAE  = 0.2800
         The average absolute prediction error is 0.28 units.

    MSE  = 0.1225
         (RMSE^2 — useful for gradient-based optimization)

    Returns
    -------
    str — the summary (also printed to stdout)
    """
    self._check_fitted("explain_fit")
    X, y = _validate_Xy(X, y)
    y_pred = self.predict(X)

    r2_val  = round(r2(y, y_pred), 4)
    rmse_val = round(rmse(y, y_pred), 4)
    mae_val = round(mae(y, y_pred), 4)
    mse_val = round(mse(y, y_pred), 4)

    r2_pct = round(r2_val * 100, 1)

    lines = [
        "Model fit summary",
        "-" * 30,
        f"R^2   = {r2_val}",
        f"     The model explains {r2_pct}% of the variance in y.",
        f"     (1.0 = perfect fit, 0.0 = no better than the mean)",
        "",
        f"RMSE = {rmse_val}",
        f"     On average, predictions are off by +/-{rmse_val} (same units as y).",
        "",
        f"MAE  = {mae_val}",
        f"     The average absolute prediction error is {mae_val} units.",
        "",
        f"MSE  = {mse_val}",
        f"     (RMSE squared — used internally by gradient descent)",
    ]

    output = "\n".join(lines)
    from linesight.utils.environment import _detect_environment
    if _detect_environment() == 'script':
        print(output)
    return output