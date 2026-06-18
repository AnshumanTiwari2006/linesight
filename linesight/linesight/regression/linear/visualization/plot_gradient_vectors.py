import numpy as np
import matplotlib.pyplot as plt
from linesight.utils.validators import _validate_Xy
from linesight.utils.colors import DATA_POINTS, FIT_LINE, RESIDUAL_POS, RESIDUAL_NEG
from linesight.exceptions import LineSightShapeError
from linesight.utils.viz_context import print_viz_context


def plot_gradient_vectors(self, X, y, scale: float = 0.3, display: bool = True):
    """
    Visualize the gradient contribution of each data point as an arrow.

    Each point contributes to the gradient update:
        contribution_i = (y_pred_i - y_i) * x_i   (for slope gradient)

    A point ABOVE the line (positive residual) pulls the line UP.
    A point BELOW the line (negative residual) pulls the line DOWN.
    The arrow length = magnitude of that point's gradient contribution.

    What is drawn
    -------------
    1. Scatter of data points, colored red (above line) or blue (below line)
    2. Regression line
    3. Vertical arrows from each point toward the line, scaled by residual
       - Arrow length = abs(residual) * scale
       - Arrow direction = toward the line (up if point is below, down if above)
    4. Net gradient annotation: "Net gradient pull: slope ↑ / ↓"

    Parameters
    ----------
    X : array-like, shape (n,) or (n, 1)
    y : array-like, shape (n,)
    scale : float, default 0.3
        Multiplier for arrow length. Increase if arrows are too small.
        Decrease if arrows overlap badly.
    display : bool, default True

    Raises
    ------
    LineSightShapeError
        If X has more than 1 feature. Gradient vectors require 2D space.
        Message: "plot_gradient_vectors() requires exactly 1 feature.
        Your X has {n} features. Gradient arrows live in 2D (x, y) space
        and cannot be drawn for high-dimensional inputs."

    Warns
    -----
    LineSightDataWarning if n_samples > 200:
        "plot_gradient_vectors() on {n} points may be cluttered.
        Consider passing a subset: X[:50], y[:50]"
    """
    import warnings
    from linesight.exceptions import LineSightDataWarning

    self._check_fitted("plot_gradient_vectors")
    X, y = _validate_Xy(X, y)

    if X.shape[1] != 1:
        raise LineSightShapeError(
            f"plot_gradient_vectors() requires exactly 1 feature.\n"
            f"Your X has {X.shape[1]} features.\n"
            f"Gradient arrows live in 2D (x, y) space and cannot be shown "
            f"for high-dimensional inputs."
        )

    if X.shape[0] > 200:
        warnings.warn(
            f"plot_gradient_vectors() on {X.shape[0]} samples may be cluttered.\n"
            f"Consider passing a subset: model.plot_gradient_vectors(X[:50], y[:50])",
            LineSightDataWarning, stacklevel=2
        )

    x = X.ravel()
    y_pred = self.predict(X)
    residuals = y_pred - y  # positive = point is below line, arrow goes up

    # Net slope gradient (sum of residual * x contributions)
    n = len(x)
    net_slope_grad = (2 / n) * np.sum(residuals * x)
    net_intercept_grad = (2 / n) * np.sum(residuals)

    fig, ax = plt.subplots(figsize=(9, 5))

    # Regression line
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = self.coef_ * x_line + self.intercept_
    ax.plot(x_line, y_line, color=FIT_LINE, linewidth=2, zorder=2, label="Fit")

    # Draw arrows and points
    for xi, yi, ypi, ri in zip(x, y, y_pred, residuals):
        color = RESIDUAL_POS if ri < 0 else RESIDUAL_NEG  # point above = red
        arrow_dy = -ri * scale  # arrow points FROM point TOWARD line
        ax.annotate(
            "", xy=(xi, yi + arrow_dy), xytext=(xi, yi),
            arrowprops=dict(
                arrowstyle="->",
                color=color,
                lw=1.2,
                alpha=0.65,
            )
        )
        ax.scatter([xi], [yi], color=color, s=45, zorder=3, edgecolors="white", lw=0.7)

    # Net gradient direction annotation
    slope_dir = "↑ (slope will increase)" if net_slope_grad < 0 else "↓ (slope will decrease)"
    ax.text(
        0.02, 0.97,
        f"Net gradient — slope: {round(net_slope_grad, 4)} {slope_dir}\n"
        f"Intercept gradient: {round(net_intercept_grad, 4)}",
        transform=ax.transAxes, fontsize=9,
        verticalalignment='top', color="#333333",
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#f5f5f5', alpha=0.8)
    )

    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.set_title("Gradient vectors — each point pulls the line toward itself")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)

    print_viz_context(
        diagram="Gradient Contribution Arrows",
        solves="Shows exactly HOW each data point is pushing the regression line during training.",
        theory=(
            "In gradient descent, each data point contributes a pull on the slope and intercept."
            " The gradient of MSE with respect to slope m is the SUM of (residual_i * x_i)"
            " across all points. Points with large residuals or large x values dominate."
            " This diagram makes that invisible math visible."
        ),
        formula="dJ/dm = (2/n) * sum((y_hat_i - y_i) * x_i)   |   dJ/db = (2/n) * sum(y_hat_i - y_i)",
        constraints=(
            "- X must have exactly 1 feature (2-D arrows)\n"
            "- Works best with <= 100 samples (more = cluttered arrows)\n"
            "- Arrow scale is approximate; only direction and relative length matter"
        ),
        reading=(
            "- Red arrow = point is ABOVE the line (model under-predicted). Pulls slope UP.\n"
            "- Blue arrow = point is BELOW the line (model over-predicted). Pulls slope DOWN.\n"
            "- Long arrow = large residual = strong gradient pull on this step.\n"
            "- Net gradient shown in top-left corner: direction the model will update next."
        ),
    )
    if display:
        return self.show(fig=fig)
    return fig
