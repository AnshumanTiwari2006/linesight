import numpy as np
import matplotlib.pyplot as plt
from linesight.utils.validators import _validate_Xy
from linesight.utils.colors import DATA_POINTS, FIT_LINE, RESIDUAL_POS


def plot_actual_vs_predicted(self, X, y, display: bool = True):
    """
    Scatter plot of actual y values vs predicted ŷ values.

    What is drawn
    -------------
    - X-axis: actual y values
    - Y-axis: predicted ŷ values
    - Diagonal line y=x (the "perfect prediction" line)
    - Each point is one sample
    - Points above the diagonal = model overestimated
    - Points below the diagonal = model underestimated
    - R² shown in corner

    Why this matters
    ----------------
    This works for ALL regression types regardless of number of features,
    because both axes are scalar (y and ŷ). It is the universal fit diagnostic.
    plot_fit() only works for 1-feature models. This one always works.

    A tight cloud around the diagonal = good fit.
    A curved cloud = the relationship is non-linear (try polynomial).
    Funnel shape = heteroscedasticity.
    """
    self._check_fitted("plot_actual_vs_predicted")
    X, y = _validate_Xy(X, y)
    y_pred = self.predict(X)

    from linesight.metrics import r2
    r2_val = round(r2(y, y_pred), 4)

    combined_min = min(y.min(), y_pred.min())
    combined_max = max(y.max(), y_pred.max())
    margin = (combined_max - combined_min) * 0.05

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(y, y_pred, color=DATA_POINTS, edgecolors="white",
               linewidths=0.8, s=60, alpha=0.8, zorder=3)

    # Perfect prediction line
    line_range = [combined_min - margin, combined_max + margin]
    ax.plot(line_range, line_range, color=FIT_LINE, linewidth=1.5,
            linestyle="--", label="Perfect prediction")

    ax.set_xlim(combined_min - margin, combined_max + margin)
    ax.set_ylim(combined_min - margin, combined_max + margin)
    ax.set_xlabel("Actual y")
    ax.set_ylabel("Predicted ŷ")
    ax.set_title(f"Actual vs Predicted  (R² = {r2_val})")
    ax.legend()
    ax.set_aspect("equal")
    ax.spines[["top", "right"]].set_visible(False)

    ax.text(0.05, 0.95,
            f"R² = {r2_val}",
            transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            color=RESIDUAL_POS if r2_val < 0.7 else FIT_LINE)

    if display:
        return self.show(fig=fig)
    return fig
