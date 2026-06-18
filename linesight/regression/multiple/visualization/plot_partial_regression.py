import numpy as np
import matplotlib.pyplot as plt
from linesight.utils.validators import _validate_Xy
from linesight.utils.viz_context import print_viz_context


def plot_partial_regression(self, X, y, feature_idx: int = 0, display: bool = True):
    """
    Marginal effect of one feature while ALL others are held at their training mean.
    The held-at values are shown in the plot subtitle.
    """
    self._check_fitted("plot_partial_regression")
    X, y = _validate_Xy(X, y)

    feature_name = self.feature_names_in_[feature_idx]
    feature_means = X.mean(axis=0)

    x_range = np.linspace(X[:, feature_idx].min(), X[:, feature_idx].max(), 200)
    X_partial = np.tile(feature_means, (200, 1))
    X_partial[:, feature_idx] = x_range
    y_partial = self.predict(X_partial)

    # Build subtitle listing held values
    held_parts = [
        f"{self.feature_names_in_[i]}={feature_means[i]:.2f}"
        for i in range(X.shape[1]) if i != feature_idx
    ]
    subtitle = "Others held at mean: " + ", ".join(held_parts) if held_parts else ""

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(X[:, feature_idx], y, color="#888888", edgecolors="white",
               s=50, alpha=0.6, zorder=3, label="Actual data")
    ax.plot(x_range, y_partial, color="#1a6fcc", linewidth=2,
            label=f"Marginal effect of \'{feature_name}\'")
    ax.set_xlabel(feature_name)
    ax.set_ylabel("Predicted y")
    ax.set_title(f"Partial Regression — \'{feature_name}\'\n{subtitle}", fontsize=11)
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    print_viz_context(
        diagram="Partial Regression (Marginal Effect) Plot",
        solves="Isolates the effect of ONE feature on the prediction while holding all others constant.",
        theory=(
            "In a multi-feature model, you cannot simply plot feature vs y because other"
            " features are also varying. Partial regression fixes all other features at"
            " their training mean and sweeps only the chosen feature across its range."
            " The resulting line shows EXACTLY the coefficient's effect in isolation."
        ),
        formula=(
            "y_partial(x_j) = b + w_j*x_j + sum(w_i * mean(x_i)) for i != j\n"
            "The slope of this line = w_j (the coefficient for feature j)"
        ),
        constraints=(
            "- All other features are held at their TRAINING mean (not test mean)\n"
            "- Only linear effects shown; cannot show interaction terms\n"
            "- feature_idx must be a valid index within X.shape[1]"
        ),
        reading=(
            "- Grey scatter = actual data (all features varying simultaneously).\n"
            "- Blue line = the pure isolated effect of this one feature.\n"
            "- Slope of blue line = the feature's coefficient.\n"
            "- Subtitle shows what values all other features are held at."
        ),
    )
    return self.show(fig=fig) if display else fig
