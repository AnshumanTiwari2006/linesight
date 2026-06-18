import numpy as np
import matplotlib.pyplot as plt
from linesight.utils.validators import _validate_Xy
from linesight.exceptions import LineSightDataWarning
import warnings
from linesight.utils.viz_context import print_viz_context


def plot_multicollinearity(self, X, display: bool = True):
    """
    Diagnose multicollinearity using a correlation heatmap + VIF scores.

    Two panels
    ----------
    Left: Correlation heatmap
        - Pearson correlation between every pair of features
        - Cells with |r| > 0.85 are outlined in red (dangerous)
        - Diagonal is 1.0 (self-correlation, always white)

    Right: VIF bar chart
        VIF (Variance Inflation Factor) for each feature.
        Formula: VIF_j = 1 / (1 - R²_j)
        where R²_j = R² from regressing feature j on all OTHER features.

        Interpretation:
        VIF = 1.0   : No correlation with other features (ideal)
        VIF = 1–5   : Acceptable
        VIF = 5–10  : Concerning — coefficients may be unstable
        VIF > 10    : Dangerous — coefficients are unreliable
        VIF → ∞     : Perfect multicollinearity (feature is linear combo of others)

        VIF > 10 bars colored red. VIF 5–10 orange. Below 5 green.

    What this teaches
    -----------------
    When two features are highly correlated, the model cannot distinguish
    their individual effects. The coefficients become unstable — a tiny
    change in data can flip them dramatically. VIF quantifies this danger.

    Parameters
    ----------
    X : array-like, shape (n, p) — p must be >= 2
    display : bool, default True

    Raises
    ------
    LineSightShapeError
        If p < 2:
        "plot_multicollinearity() requires at least 2 features.
        Your X has only 1 feature. Multicollinearity is a multi-feature problem."
    """
    from linesight.utils.validators import _validate_X
    from linesight.exceptions import LineSightShapeError

    self._check_fitted("plot_multicollinearity")
    X = _validate_X(X)
    n, p = X.shape

    if p < 2:
        raise LineSightShapeError(
            f"plot_multicollinearity() requires at least 2 features.\n"
            f"Your X has only 1 feature.\n"
            f"Multicollinearity is a multi-feature problem."
        )

    feature_names = getattr(self, 'feature_names_in_',
                            [f"x{j}" for j in range(p)])

    # Pearson correlation matrix
    corr = np.corrcoef(X.T)

    # VIF for each feature: R² from regressing feature j on all others
    from linesight.metrics import r2 as r2_metric
    vifs = []
    for j in range(p):
        X_others = np.delete(X, j, axis=1)
        # Use normal equations for VIF (fast, exact)
        X_bias = np.hstack([np.ones((n, 1)), X_others])
        try:
            theta = np.linalg.lstsq(X_bias, X[:, j], rcond=None)[0]
            y_hat_j = X_bias @ theta
            r2_j = r2_metric(X[:, j], y_hat_j)
            r2_j = min(r2_j, 0.9999)  # cap to avoid VIF → ∞ in display
            vif_j = 1.0 / max(1.0 - r2_j, 1e-6)
        except Exception:
            vif_j = 0.0
        vifs.append(vif_j)

    if any(v > 10 for v in vifs):
        warnings.warn(
            f"High multicollinearity detected. Features with VIF > 10: "
            f"{[feature_names[j] for j, v in enumerate(vifs) if v > 10]}.\n"
            f"Coefficients for these features are unreliable. "
            f"Consider removing or combining correlated features.",
            LineSightDataWarning, stacklevel=2
        )

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, max(4, p * 0.7)))

    # Left: correlation heatmap
    im = ax_left.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax_left.set_xticks(range(p))
    ax_left.set_yticks(range(p))
    ax_left.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=8)
    ax_left.set_yticklabels(feature_names, fontsize=8)

    # Annotate cells with correlation value
    for i in range(p):
        for j in range(p):
            val = round(corr[i, j], 2)
            text_color = "white" if abs(val) > 0.6 else "black"
            ax_left.text(j, i, str(val), ha='center', va='center',
                         fontsize=7, color=text_color)
            # Red outline for dangerous correlation
            if i != j and abs(corr[i, j]) > 0.85:
                rect = plt.Rectangle([j - 0.5, i - 0.5], 1, 1,
                                     fill=False, edgecolor='red', linewidth=2)
                ax_left.add_patch(rect)

    plt.colorbar(im, ax=ax_left, shrink=0.8, label="Pearson correlation")
    ax_left.set_title("Feature correlation matrix\n(red outline = |r| > 0.85, dangerous)")

    # Right: VIF bar chart
    vif_colors = []
    for v in vifs:
        if v > 10:
            vif_colors.append("#e05252")
        elif v > 5:
            vif_colors.append("#e08c2a")
        else:
            vif_colors.append("#2a9d4e")

    bars = ax_right.barh(feature_names, vifs, color=vif_colors, alpha=0.85)
    ax_right.axvline(x=5, color="#e08c2a", linewidth=1.2, linestyle="--",
                     alpha=0.7, label="VIF=5 (concerning)")
    ax_right.axvline(x=10, color="#e05252", linewidth=1.5, linestyle="--",
                     alpha=0.9, label="VIF=10 (dangerous)")

    for bar, vif in zip(bars, vifs):
        ax_right.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                      f"  {round(vif, 1)}", va='center', fontsize=9)

    ax_right.set_xlabel("VIF score")
    ax_right.set_title("Variance Inflation Factor (VIF)\nGreen<5 · Orange 5–10 · Red>10")
    ax_right.legend(fontsize=8)
    ax_right.spines[["top", "right"]].set_visible(False)

    plt.suptitle("Multicollinearity diagnosis", fontsize=12)
    plt.tight_layout()

    print_viz_context(
        diagram="Multicollinearity Diagnosis (Correlation Heatmap + VIF)",
        solves="Detects when two or more features are so correlated that coefficients become unreliable.",
        theory=(
            "When feature A is nearly a linear combination of feature B, the model cannot"
            " determine how much of the prediction is due to A vs B. The coefficients"
            " become unstable — tiny data changes cause huge coefficient swings."
            " VIF measures this: it fits each feature as a regression TARGET using all"
            " other features as predictors, then checks how well it can be predicted."
        ),
        formula=(
            "VIF_j = 1 / (1 - R2_j)\n"
            "where R2_j = R-squared from regressing feature j on all other features\n"
            "Pearson r_ij = cov(x_i, x_j) / (std(x_i) * std(x_j))"
        ),
        constraints=(
            "- Requires at least 2 features\n"
            "- VIF > 10 means coefficients are unreliable (not that the model is wrong)\n"
            "- Perfect multicollinearity (VIF = inf) makes the model unsolvable"
        ),
        reading=(
            "- Left heatmap: red cells = high positive correlation, blue = negative.\n"
            "- Red-outlined cell = |r| > 0.85 (dangerously correlated pair).\n"
            "- Right bars: Green VIF < 5 (safe), Orange 5-10 (watch), Red > 10 (remove one).\n"
            "- Fix: drop one of the correlated features, or use Ridge/Lasso regularization."
        ),
    )
    if display:
        return self.show(fig=fig)
    return fig
