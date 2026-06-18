import numpy as np
import matplotlib.pyplot as plt
from linesight.utils.validators import _validate_Xy
from linesight.utils.colors import DATA_POINTS, FIT_LINE, RESIDUAL_POS, WARNING_COLOR
from linesight.exceptions import LineSightShapeError
import warnings
from linesight.exceptions import LineSightDataWarning
from linesight.utils.viz_context import print_viz_context


def plot_sensitivity_analysis(self, X, y,
                               top_n: int = 3,
                               display: bool = True):
    """
    Show how the regression line changes when each point is removed.

    For each data point i, compute:
    1. Fit a temporary model on all points EXCEPT i
    2. Record the new slope and intercept
    3. Plot this "leave-one-out" line in light gray
    4. Highlight the top_n most influential points (those that, when removed,
       change the slope the most) in red

    What is drawn
    -------------
    Left subplot: scatter + regression lines
    - All "leave-one-out" lines overlaid in light gray (alpha=0.15)
    - The original fitted line in blue
    - Top_n most influential points highlighted with red circles and labeled
    - A "stability band": if all gray lines stay close to the blue, the model
      is stable. If some gray lines deviate far, there are influential points.

    Right subplot: influence score bar chart
    - X-axis: sample index
    - Y-axis: abs(slope_change) when that point is removed
    - Top_n bars colored red

    Parameters
    ----------
    X : array-like, shape (n, 1)
    y : array-like, shape (n,)
    top_n : int, default 3
        Number of most influential points to highlight.
    display : bool, default True

    Raises
    ------
    LineSightShapeError
        If X has more than 1 feature:
        "plot_sensitivity_analysis() requires single-feature X.
        Your X has {n} features. Leave-one-out lines require 2D visualization."

    LineSightShapeError
        If n < 5:
        "plot_sensitivity_analysis() requires at least 5 samples.
        Your dataset has only {n} samples. Leave-one-out fitting is
        unreliable with very small datasets."

    Warns
    -----
    LineSightDataWarning if n > 500:
        "Fitting {n} leave-one-out models may be slow.
        Consider passing a subset: X[:100], y[:100]"
    """
    self._check_fitted("plot_sensitivity_analysis")
    X, y = _validate_Xy(X, y)

    if X.shape[1] != 1:
        raise LineSightShapeError(
            f"plot_sensitivity_analysis() requires single-feature X.\n"
            f"Your X has {X.shape[1]} features.\n"
            f"Leave-one-out lines require 2D (x, y) visualization."
        )

    n = X.shape[0]
    if n < 5:
        raise LineSightShapeError(
            f"plot_sensitivity_analysis() requires at least 5 samples.\n"
            f"Your dataset has only {n} samples.\n"
            f"Leave-one-out fitting is unreliable with very small datasets."
        )

    if n > 500:
        warnings.warn(
            f"Fitting {n} leave-one-out models may be slow.\n"
            f"Consider: model.plot_sensitivity_analysis(X[:100], y[:100])",
            LineSightDataWarning, stacklevel=2
        )

    x = X.ravel()
    original_slope = float(np.ravel(self.coef_)[0])
    x_line = np.linspace(x.min(), x.max(), 200)

    # Reuse same class with same hyperparameters
    ModelClass = self.__class__

    loo_slopes = []
    loo_intercepts = []

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_loo = X[mask]
        y_loo = y[mask]

        tmp = ModelClass(
            learning_rate=self.learning_rate,
            epochs=self.epochs,
        )
        tmp.fit(X_loo, y_loo)
        loo_slopes.append(float(np.ravel(tmp.coef_)[0]))
        loo_intercepts.append(float(tmp.intercept_))

    slope_changes = np.abs(np.array(loo_slopes) - original_slope)
    influence_order = np.argsort(slope_changes)[::-1]
    top_influential = influence_order[:top_n]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: regression lines
    for i, (m, b) in enumerate(zip(loo_slopes, loo_intercepts)):
        color = WARNING_COLOR if i in top_influential else "#aaaaaa"
        alpha = 0.6 if i in top_influential else 0.12
        lw = 1.5 if i in top_influential else 0.8
        ax1.plot(x_line, m * x_line + b, color=color, alpha=alpha, linewidth=lw)

    # Original line on top
    y_orig_line = original_slope * x_line + float(self.intercept_)
    ax1.plot(x_line, y_orig_line, color=FIT_LINE, linewidth=2.5,
             label="Original fit", zorder=5)

    # Data points
    ax1.scatter(x, y, color=DATA_POINTS, edgecolors="white",
                s=50, linewidths=0.8, zorder=4)

    # Highlight influential points
    for i in top_influential:
        ax1.scatter([x[i]], [y[i]], color=WARNING_COLOR, s=120,
                    edgecolors="white", linewidths=1.5, zorder=6)
        ax1.annotate(
            f"  #{i}\n  Δslope={round(slope_changes[i], 3)}",
            xy=(x[i], y[i]), fontsize=8, color=WARNING_COLOR
        )

    ax1.set_title("Leave-one-out regression lines\n"
                  "Red lines deviate most from original (= influential points)")
    ax1.set_xlabel("X")
    ax1.set_ylabel("y")
    ax1.legend()
    ax1.spines[["top", "right"]].set_visible(False)

    # Right: influence bar chart
    bar_colors = [WARNING_COLOR if i in top_influential else "#aaaaaa"
                  for i in range(n)]
    ax2.bar(range(n), slope_changes, color=bar_colors, alpha=0.8)
    ax2.set_xlabel("Sample index")
    ax2.set_ylabel("|Slope change when removed|")
    ax2.set_title(f"Influence score per sample\n"
                  f"Top {top_n} most influential highlighted in red")
    ax2.spines[["top", "right"]].set_visible(False)

    plt.suptitle("Sensitivity analysis — how much does each point affect the model?",
                 fontsize=12)
    plt.tight_layout()

    print_viz_context(
        diagram="Leave-One-Out Sensitivity Analysis",
        solves="Identifies which individual data points have the most POWER to change the model.",
        theory=(
            "For each data point i, a temporary model is trained on ALL OTHER points"
            " (leave-one-out). The slope of that model is compared to the original."
            " If removing point i causes a large change in slope, that point is 'influential'."
            " Influential points are not necessarily outliers — a perfectly on-line"
            " extreme point can be very influential without being an error."
        ),
        formula=(
            "Influence_i = | slope_original - slope_without_i |\n"
            "For each i: train on X[mask], y[mask] where mask[i] = False"
        ),
        constraints=(
            "- X must have exactly 1 feature\n"
            "- Requires at least 5 samples to be meaningful\n"
            "- Runs N separate fits (slow for N > 500; use X[:100] for large datasets)\n"
            "- Shows leave-one-out slope change only, not Cook's Distance"
        ),
        reading=(
            "- Grey lines = what the fit would look like without that one point.\n"
            "- Red lines = the most influential points (deviate most from original).\n"
            "- Bar chart (right) = influence score per sample index.\n"
            "- Tall red bars = investigate those rows; they may be errors or outliers."
        ),
    )
    if display:
        return self.show(fig=fig)
    return fig
