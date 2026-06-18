import numpy as np
import matplotlib.pyplot as plt
from linesight.utils.validators import _validate_Xy
from linesight.utils.colors import COLOR_LASSO, RESIDUAL_POS
from linesight.utils.viz_context import print_viz_context


def plot_sparsity_path(self, X, y, alphas=None, display: bool = True):
    """
    Plot the number of nonzero features as alpha increases.

    Shows Lasso's feature selection behavior as a step function:
    each step-down = one more feature eliminated.

    What is drawn
    -------------
    X-axis: alpha (log scale)
    Y-axis: number of features with |coefficient| > 1e-6 (nonzero)
    Step-function line showing how many features survive at each alpha
    Vertical dashed line at current self.alpha
    Annotations: which feature is eliminated at each step

    Parameters
    ----------
    X : array-like
    y : array-like
    alphas : list of float, optional
        Default: 60 log-spaced values from 1e-4 to 1e1
    display : bool, default True
    """
    self._check_fitted("plot_sparsity_path")
    X, y = _validate_Xy(X, y)
    p = X.shape[1]

    if alphas is None:
        alphas = np.logspace(-4, 1, 60)

    feature_names = getattr(self, 'feature_names_in_',
                            [f"x{j}" for j in range(p)])

    from linesight.regression.lasso.core import LassoRegression

    nonzero_counts = []
    coef_arrays = []

    for alpha in alphas:
        tmp = LassoRegression(alpha=alpha, epochs=self.epochs)
        tmp.fit(X, y)
        coefs = tmp.weights
        nonzero_counts.append(int(np.sum(np.abs(coefs) > 1e-6)))
        coef_arrays.append(coefs.copy())

    # Find elimination events: where count drops
    elimination_alphas = []
    elimination_labels = []

    prev_active = set(range(p))
    for k, (alpha, coefs) in enumerate(zip(alphas, coef_arrays)):
        current_active = set(j for j in range(p) if abs(coefs[j]) > 1e-6)
        eliminated = prev_active - current_active
        for j in eliminated:
            elimination_alphas.append(alpha)
            elimination_labels.append(feature_names[j])
        prev_active = current_active

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.step(alphas, nonzero_counts, where='post', color=COLOR_LASSO, linewidth=2)
    ax.axvline(self.alpha, color=RESIDUAL_POS, linewidth=1.5,
               linestyle="--", label=f"Current α = {self.alpha}")

    # Annotate elimination events (avoid overlap by alternating y position)
    annotated = {}
    for alpha, label in zip(elimination_alphas, elimination_labels):
        if label not in annotated:
            y_offset = 0.5
            ax.annotate(
                f"'{label}' → 0",
                xy=(alpha, nonzero_counts[np.argmin(np.abs(alphas - alpha))]),
                xytext=(alpha, nonzero_counts[np.argmin(np.abs(alphas - alpha))] + y_offset),
                fontsize=7, color="#555555",
                arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.8)
            )
            annotated[label] = True

    ax.set_xscale('log')
    ax.set_xlabel("Alpha (regularization strength) — log scale")
    ax.set_ylabel("Number of nonzero features")
    ax.set_title(
        f"Lasso sparsity path — {p} features, shrinking to 0 as alpha increases\n"
        f"Each step = one feature eliminated"
    )
    ax.set_ylim(-0.5, p + 0.5)
    ax.set_yticks(range(p + 1))
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)

    print_viz_context(
        diagram="Lasso Sparsity Path (Feature Count vs Alpha)",
        solves="Shows at exactly WHICH alpha value each feature gets eliminated from the model.",
        theory=(
            "Lasso uses an L1 penalty: alpha * sum(|w_j|).  Unlike Ridge,"
            " Lasso can drive weights to EXACTLY zero via soft-thresholding."
            " The soft-threshold operator sets w_j = 0 whenever the residual"
            " correlation |rho_j| < alpha.  As alpha increases, more features"
            " fall below this threshold and are eliminated entirely."
        ),
        formula=(
            "Lasso: min  MSE + alpha * sum(|w_j|)\n"
            "Soft-threshold: w_j = sign(rho_j) * max(|rho_j| - alpha, 0)\n"
            "where rho_j = (1/n) * X_j' * (y - y_excl_j)   (partial residual)"
        ),
        constraints=(
            "- alpha must be >= 0\n"
            "- Features must be on the same scale for fair elimination order\n"
            "- Step function shape is characteristic of Lasso (not Ridge)"
        ),
        reading=(
            "- Y-axis = how many features the model is USING at each alpha.\n"
            "- Each downward step = one more feature eliminated (coefficient -> 0).\n"
            "- Annotations show which feature is eliminated and at what alpha.\n"
            "- Your chosen alpha (dashed line) shows the current level of sparsity."
        ),
    )
    if display:
        return self.show(fig=fig)
    return fig
