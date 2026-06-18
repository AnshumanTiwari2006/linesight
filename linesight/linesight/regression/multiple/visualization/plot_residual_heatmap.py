import numpy as np
import matplotlib.pyplot as plt
from linesight.utils.validators import _validate_Xy
from linesight.exceptions import LineSightDataWarning
import warnings


def plot_residual_heatmap(self, X, y, display: bool = True):
    """
    Heatmap of residuals vs features: reveals which feature ranges cause errors.

    Layout
    ------
    Top panel: Residual bar chart (per sample, sorted by residual magnitude)
    Bottom panel: Feature value heatmap (same sample order)
       - Each row = one feature
       - Each column = one sample (sorted by residual)
       - Color = z-scored feature value (high = dark, low = light)

    If prediction errors cluster in columns where a feature is high/low,
    that feature is systematically related to the error — a signal that
    the model needs that feature transformed or a new interaction term added.

    Parameters
    ----------
    X : array-like, shape (n, p)
    y : array-like, shape (n,)
    display : bool, default True

    Warns
    -----
    LineSightDataWarning if p > 20:
        "Your model has {p} features. The heatmap may be unreadable.
        Consider visualizing subsets of features."
    """
    self._check_fitted("plot_residual_heatmap")
    X, y = _validate_Xy(X, y)
    p = X.shape[1]

    if p > 20:
        warnings.warn(
            f"Your model has {p} features. The heatmap may be unreadable.\n"
            f"Consider: model.plot_residual_heatmap(X[:, :10], y)",
            LineSightDataWarning, stacklevel=2
        )

    y_pred = self.predict(X)
    residuals = y - y_pred

    # Sort samples by residual magnitude for cleaner pattern visibility
    sort_order = np.argsort(residuals)
    residuals_sorted = residuals[sort_order]
    X_sorted = X[sort_order]

    feature_names = getattr(self, 'feature_names_in_',
                            [f"x{j}" for j in range(p)])

    # Z-score each feature for comparable color scale
    X_z = (X_sorted - X_sorted.mean(axis=0)) / (X_sorted.std(axis=0) + 1e-10)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(min(14, max(8, X.shape[0] * 0.15)), 6),
        gridspec_kw={'height_ratios': [1, 2]}, sharex=True
    )

    # Top: residual bars
    colors = ["#e05252" if r > 0 else "#5280e0" for r in residuals_sorted]
    ax_top.bar(range(len(residuals_sorted)), residuals_sorted, color=colors, alpha=0.8)
    ax_top.axhline(0, color="#333333", linewidth=0.8)
    ax_top.set_ylabel("Residual\n(y − ŷ)")
    ax_top.set_title("Residuals (sorted) vs Feature values\n"
                     "Patterns in the heatmap reveal which features cause errors")
    ax_top.spines[["top", "right"]].set_visible(False)

    # Bottom: feature heatmap
    im = ax_bot.imshow(X_z.T, aspect='auto', cmap='RdBu_r', vmin=-3, vmax=3)
    ax_bot.set_yticks(range(p))
    ax_bot.set_yticklabels(feature_names, fontsize=8)
    ax_bot.set_xlabel("Sample (sorted by residual: underpredicted → overpredicted)")
    ax_bot.set_ylabel("Feature")

    plt.colorbar(im, ax=ax_bot, label="Feature value (z-score)", shrink=0.8)
    plt.tight_layout()

    if display:
        return self.show(fig=fig)
    return fig
