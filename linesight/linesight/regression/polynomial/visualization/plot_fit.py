import numpy as np
import matplotlib.pyplot as plt
from linesight.utils.validators import _validate_Xy
from linesight.utils.colors import DATA_POINTS, FIT_LINE, RESIDUAL_POS, RESIDUAL_NEG


def plot_fit(self, X, y, show_residuals: bool = True, display: bool = True):
    """
    Plot raw data, the fitted polynomial curve, and optional residuals.

    Key difference from LinearRegression.plot_fit
    ----------------------------------------------
    The fit line is a SMOOTH CURVE, not a straight line.
    We generate 300 evenly-spaced x values from min to max and predict on them.
    This produces the smooth polynomial curve the model has learned.

    What is drawn
    -------------
    1. Scatter of data points
    2. Smooth polynomial curve (300 points for smoothness)
    3. If show_residuals=True: vertical dashed bars from each point to the
       NEAREST POINT ON THE CURVE (not to the line, since there is no line).
       These bars are approximated by finding the curve's y at each data x.
    4. Degree annotation in corner: "Degree 3 polynomial"
    """
    self._check_fitted("plot_fit")
    X, y = _validate_Xy(X, y)

    x = X.ravel()

    # Smooth curve
    x_curve = np.linspace(x.min(), x.max(), 300)
    y_curve = self.predict(x_curve.reshape(-1, 1))

    # Predictions at actual data x values (for residual bars)
    y_pred = self.predict(X)

    fig, ax = plt.subplots(figsize=(8, 5))

    if show_residuals:
        for xi, yi, ypi in zip(x, y, y_pred):
            color = RESIDUAL_POS if yi > ypi else RESIDUAL_NEG
            ax.plot([xi, xi], [yi, ypi], color=color,
                    linewidth=1, linestyle="--", alpha=0.6)

    ax.scatter(x, y, color=DATA_POINTS, edgecolors="white",
               linewidths=0.8, s=60, zorder=3, label="Data")
    ax.plot(x_curve, y_curve, color=FIT_LINE, linewidth=2,
            label=f"Degree {self.degree} fit")

    ax.set_title(f"Polynomial Regression — Degree {self.degree}", fontsize=13)
    ax.set_xlabel("X")
    ax.set_ylabel("ŷ")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)

    if display:
        return self.show(fig=fig)
    return fig
