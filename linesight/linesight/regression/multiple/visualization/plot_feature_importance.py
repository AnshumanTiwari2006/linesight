import numpy as np
import matplotlib.pyplot as plt
from linesight.utils.viz_context import print_viz_context


def plot_feature_importance(self, display: bool = True):
    """
    Bar chart of |coefficient| for each feature (excluding intercept), sorted descending.
    Horizontal line at mean importance for reference.
    """
    self._check_fitted("plot_feature_importance")

    coefs = self.weights
    importances = np.abs(coefs)
    names = self.feature_names_in_

    sorted_idx = np.argsort(importances)[::-1]
    sorted_names = [names[i] for i in sorted_idx]
    sorted_vals = importances[sorted_idx]

    fig, ax = plt.subplots(figsize=(max(6, len(names)), 5))
    bars = ax.bar(range(len(names)), sorted_vals, color="#1a6fcc", alpha=0.8, edgecolor="white")
    ax.axhline(sorted_vals.mean(), color="#e05252", linestyle="--",
               linewidth=1, label=f"Mean = {round(sorted_vals.mean(), 3)}")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(sorted_names, rotation=30, ha='right')
    ax.set_ylabel("|Coefficient|")
    ax.set_title("Feature Importance (by coefficient magnitude)")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    print_viz_context(
        diagram="Feature Importance Bar Chart",
        solves="Ranks which features the model is relying on most to make predictions.",
        theory=(
            "In multiple linear regression, each feature gets a weight (coefficient)."
            " The magnitude (absolute value) of that weight tells us how much a"
            " 1-unit change in that feature shifts the prediction. Larger magnitude"
            " = the model trusts that feature more. NOTE: this only works fairly"
            " when features are on the same scale (use normalize=True)."
        ),
        formula="y = w0 + w1*x1 + w2*x2 + ... + wn*xn   |   importance_i = |w_i|",
        constraints=(
            "- Only meaningful when features are on comparable scales (normalize=True)\n"
            "- Does NOT account for correlation between features\n"
            "- A small coefficient does NOT mean the feature is unimportant if features are unscaled"
        ),
        reading=(
            "- Tallest bar = feature the model weights most heavily.\n"
            "- Dashed red line = average importance across all features.\n"
            "- Features below average may be candidates for removal.\n"
            "- For reliable ranking, always fit with normalize=True."
        ),
    )
    if display:
        return self.show(fig=fig)
    return fig
