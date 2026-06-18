import numpy as np
import matplotlib.pyplot as plt
from linesight.utils.colors import FIT_LINE, CLASS_0, CLASS_1


def plot_log_odds(self, X, y, display: bool = True):
    """
    Show the three representations of logistic regression output.

    Three panels, all sharing the same x-axis (sample index sorted by z):
    -----------------------------------------------------------------------
    Panel 1: Linear score z = Xθ + b
        - Shows raw model output before sigmoid
        - Unbounded: can be any real number
        - Dashed line at z=0 (the decision threshold)

    Panel 2: Probability p = σ(z)
        - The sigmoid-transformed output
        - Bounded [0, 1]
        - Dashed line at p=0.5
        - Colored by TRUE class (red=class 1, blue=class 0)
        - Misclassified points marked with ×

    Panel 3: Log-odds = log(p / (1-p)) = z
        - Shows that log-odds IS the linear score (they are identical)
        - Dashed line at log-odds=0

    The key insight: panels 1 and 3 are IDENTICAL because log(σ(z)/(1-σ(z))) = z.
    The title states this explicitly.

    Parameters
    ----------
    X : array-like, shape (n, p)
    y : array-like, shape (n,)
    display : bool, default True

    Warns
    -----
    LineSightDataWarning if n > 500:
        "plot_log_odds() on {n} samples may be slow to render.
        Consider plotting a subset."
    """
    import warnings
    from linesight.exceptions import LineSightDataWarning
    from linesight.utils.validators import _validate_Xy
    from linesight.utils.array_utils import _add_bias_column
    from linesight.regression.logistic.engine.sigmoid import _sigmoid

    self._check_fitted("plot_log_odds")
    X, y = _validate_Xy(X, y)
    n = X.shape[0]

    if n > 500:
        warnings.warn(
            f"plot_log_odds() on {n} samples may be slow.\n"
            f"Consider: model.plot_log_odds(X[:200], y[:200])",
            LineSightDataWarning, stacklevel=2
        )

    X_bias = _add_bias_column(X)
    z = X_bias @ np.concatenate(([self.bias], self.weights))  # linear scores
    proba = _sigmoid(z)
    proba_clipped = np.clip(proba, 1e-10, 1 - 1e-10)
    log_odds = np.log(proba_clipped / (1 - proba_clipped))

    y_pred = self.predict(X)
    misclassified = y_pred != y.astype(int)

    # Sort by z score for clean left-to-right ordering
    sort_idx = np.argsort(z)
    z_s = z[sort_idx]
    p_s = proba[sort_idx]
    lo_s = log_odds[sort_idx]
    y_s = y[sort_idx].astype(int)
    mis_s = misclassified[sort_idx]

    idx = np.arange(len(z_s))
    colors_by_class = [CLASS_1 if yi == 1 else CLASS_0 for yi in y_s]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    # Panel 1: z scores
    ax1.scatter(idx, z_s, c=colors_by_class, s=25, alpha=0.8, edgecolors='none')
    ax1.axhline(0, color="#333333", linewidth=1, linestyle="--", alpha=0.6)
    ax1.set_ylabel("z = Xθ (linear score)")
    ax1.set_title("Logistic regression output decomposition\n"
                  "Samples sorted left→right by linear score z")
    ax1.spines[["top", "right"]].set_visible(False)

    # Panel 2: probabilities
    for xi, pi, ci, mi in zip(idx, p_s, colors_by_class, mis_s):
        marker = "x" if mi else "o"
        ms = 60 if mi else 20
        ax2.scatter([xi], [pi], c=[ci], s=ms, marker=marker,
                    alpha=0.9, edgecolors='none')
    ax2.axhline(0.5, color="#333333", linewidth=1, linestyle="--", alpha=0.6)
    ax2.set_ylabel("P(class=1) = σ(z)")
    ax2.set_ylim(-0.05, 1.05)
    ax2.text(len(idx) * 0.02, 0.52, "decision threshold p=0.5",
             fontsize=8, color="#555555")
    ax2.spines[["top", "right"]].set_visible(False)

    # Custom legend for panel 2
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=CLASS_1,
               markersize=8, label='Class 1 (correct)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=CLASS_0,
               markersize=8, label='Class 0 (correct)'),
        Line2D([0], [0], marker='x', color=CLASS_1,
               markersize=8, label='Misclassified'),
    ]
    ax2.legend(handles=legend_elements, fontsize=8, loc='upper left')

    # Panel 3: log-odds
    ax3.scatter(idx, lo_s, c=colors_by_class, s=25, alpha=0.8, edgecolors='none')
    ax3.axhline(0, color="#333333", linewidth=1, linestyle="--", alpha=0.6)
    ax3.set_ylabel("log-odds = log(p/(1−p))")
    ax3.set_xlabel("Sample index (sorted by z)")
    ax3.text(0.5, 0.02,
             "Note: log-odds = z (Panels 1 and 3 are identical — this is why sigmoid works)",
             transform=ax3.transAxes, ha='center', fontsize=8, color="#555555")
    ax3.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()

    if display:
        return self.show(fig=fig)
    return fig
