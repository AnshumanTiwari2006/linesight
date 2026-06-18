import numpy as np
import matplotlib.pyplot as plt
import warnings
from linesight.utils.validators import _validate_Xy
from linesight.exceptions import LineSightShapeError
from linesight.utils.viz_context import print_viz_context


def plot_fit(self, X, y, show_residuals: bool = True, display: bool = True):
    """
    Plot the data, the fitted regression line, and optional residual bars.

    What is drawn
    -------------
    1. Scatter plot of raw data points (gray, edged in white)
    2. Regression line (blue) from min(X) to max(X)
    3. If show_residuals=True:
         - Vertical dashed lines from each point to the line
         - Red dashes = point is above the line (positive residual)
         - Blue dashes = point is below the line (negative residual)
       This teaches the student VISUALLY what residuals are.

    Parameters
    ----------
    X : array-like, shape (n,) or (n, 1) — single feature only
    y : array-like, shape (n,)
    show_residuals : bool, default True
    display : bool, default True
        If True, calls self.show(fig=fig) which handles Jupyter/script routing.
        If False, returns the figure without displaying.

    Raises
    ------
    LineSightShapeError if X has more than 1 feature. Use
    MultipleLinearRegression.plot_partial_regression() for multi-feature models.
    """
    self._check_fitted("plot_fit")
    X, y = _validate_Xy(X, y)

    if X.shape[1] != 1:
        raise LineSightShapeError(
            f"plot_fit() only works for single-feature models.\n"
            f"Your X has {X.shape[1]} features.\n"
            f"For multi-feature models, use plot_partial_regression(feature_idx=0)."
        )

    x = X.ravel()
    y_pred = self.predict(X)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Residual bars first so they appear behind the points
    if show_residuals:
        for xi, yi, ypi in zip(x, y, y_pred):
            color = "#e05252" if yi > ypi else "#5280e0"
            ax.plot([xi, xi], [yi, ypi], color=color, linewidth=1,
                    linestyle="--", alpha=0.6)

    # Data points
    ax.scatter(x, y, color="#888888", edgecolors="white", linewidths=0.8,
               s=60, zorder=3, label="Data")

    # Regression line
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = self.m * x_line + self.b
    ax.plot(x_line, y_line, color="#1a6fcc", linewidth=2, label="Fit")

    m = round(float(self.m), 3)
    b = round(float(self.b), 3)
    sign = "+" if b >= 0 else "-"
    ax.set_title(f"y = {m}x {sign} {abs(b)}", fontsize=13)
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    print_viz_context(
        diagram="Regression Fit Plot",
        solves="Shows where the learned line sits relative to the actual data points.",
        theory=(
            "Simple linear regression fits the line  y = mx + b  by minimizing"
            " the Mean Squared Error (MSE) between predicted and actual values."
            " Gradient descent iteratively adjusts m and b until the line can"
            " no longer improve."
        ),
        formula="y = m*x + b    MSE = (1/n) * sum((y_i - (m*x_i + b))^2)",
        constraints=(
            "- X must have exactly 1 feature (2-D plot)\n"
            "- Assumes a LINEAR relationship between X and y\n"
            "- Assumes errors are normally distributed around the line\n"
            "- Sensitive to outliers; a single extreme point pulls the line"
        ),
        reading=(
            "- Blue line is the model. Dashed bars are residuals (errors).\n"
            "- Red bars = model UNDER-predicted (actual > predicted).\n"
            "- Blue bars = model OVER-predicted (actual < predicted).\n"
            "- A good fit has short, evenly distributed bars with no pattern."
        ),
    )
    if display:
        return self.show(fig=fig)
    return fig
