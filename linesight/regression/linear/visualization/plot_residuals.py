import numpy as np
import matplotlib.pyplot as plt
from linesight.utils.validators import _validate_Xy
from linesight.utils.viz_context import print_viz_context


def plot_residuals(self, X, y, display: bool = True):
    """
    Plot residuals (actual - predicted) against fitted values.

    What is drawn
    -------------
    - X-axis: fitted values (y_hat)
    - Y-axis: residuals (y - y_hat)
    - Horizontal dashed line at y=0 (where a perfect model would sit)
    - Points colored red (positive residual = underprediction)
      and blue (negative residual = overprediction)

    What to look for
    ----------------
    - Random scatter around zero = good fit, assumptions met
    - Curved pattern = linear model is wrong, try polynomial
    - Funnel shape = heteroscedasticity (variance increases with fitted value)
    - Outliers far from zero = influential data points

    Parameters
    ----------
    X : array-like
    y : array-like
    display : bool, default True
    """
    self._check_fitted("plot_residuals")
    X, y = _validate_Xy(X, y)
    y_pred = self.predict(X)
    residuals = y - y_pred

    colors = ["#e05252" if r > 0 else "#5280e0" for r in residuals]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_pred, residuals, c=colors, edgecolors="white",
               linewidths=0.8, s=60, zorder=3)
    ax.axhline(0, color="#333333", linewidth=1, linestyle="--", alpha=0.5)

    ax.set_xlabel("Fitted values (ŷ)")
    ax.set_ylabel("Residuals (y − ŷ)")
    ax.set_title("Residuals vs Fitted")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    print_viz_context(
        diagram="Residuals vs Fitted Plot",
        solves="Diagnoses whether the linear model's assumptions are satisfied.",
        theory=(
            "A residual is the difference between the actual target y_i and the"
            " model's prediction  y_hat_i.  If the model is correct, residuals"
            " should be RANDOM noise with no pattern — they should scatter evenly"
            " around zero regardless of the fitted value."
        ),
        formula="Residual_i = y_i - y_hat_i = y_i - (m*x_i + b)",
        constraints=(
            "- Requires a fitted model (fit() must have been called first)\n"
            "- Assumes residuals are independent and identically distributed\n"
            "- Assumes constant variance (homoscedasticity) across fitted values"
        ),
        reading=(
            "- Random scatter around the y=0 line = model assumptions are MET.\n"
            "- Curved / U-shaped pattern = linear model is WRONG (try polynomial).\n"
            "- Funnel shape (spread widens) = heteroscedasticity problem.\n"
            "- Isolated far-off points = outliers heavily influencing the model."
        ),
    )
    if display:
        return self.show(fig=fig)
    return fig
