import numpy as np
import matplotlib.pyplot as plt
from linesight.utils.viz_context import print_viz_context


def plot_l1_l2_balance(self, X, y, l1_ratios=None, display: bool = True):
    """
    Show how coefficient behavior changes as l1_ratio shifts from Ridge to Lasso.

    l1_ratio=0.0 is pure Ridge: all coefficients shrink, none zero.
    l1_ratio=1.0 is pure Lasso: some coefficients become exactly zero.
    l1_ratio=0.5 is ElasticNet: mix.

    What is drawn
    -------------
    One subplot per l1_ratio value showing coefficient bar chart.
    Zeroed coefficients shown in gray. Non-zero in purple.
    Title shows the interpolation: "l1_ratio=0.0 (Ridge)" ... "l1_ratio=1.0 (Lasso)"

    Parameters
    ----------
    X : array-like
    y : array-like
    l1_ratios : list of float, optional
        Default: [0.0, 0.2, 0.5, 0.8, 1.0]
    display : bool, default True
    """
    self._check_fitted("plot_l1_l2_balance")

    from linesight.utils.validators import _validate_Xy
    from linesight.regression.elasticnet.core import ElasticNetRegression

    X, y = _validate_Xy(X, y)

    if l1_ratios is None:
        l1_ratios = [0.0, 0.2, 0.5, 0.8, 1.0]

    feature_names = getattr(self, 'feature_names_in_',
                            [f"x{j}" for j in range(X.shape[1])])

    all_coefs = []
    for ratio in l1_ratios:
        tmp = ElasticNetRegression(
            alpha=self.alpha, l1_ratio=ratio,
            epochs=self.epochs
        )
        tmp.fit(X, y)
        all_coefs.append(tmp.weights)

    max_abs = max(max(abs(c).max() for c in all_coefs), 0.01) * 1.15

    n = len(l1_ratios)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for i, (ratio, coefs) in enumerate(zip(l1_ratios, all_coefs)):
        ax = axes[i]
        bar_colors = ["#7047c4" if abs(c) > 1e-10 else "#cccccc" for c in coefs]
        ax.bar(feature_names, coefs, color=bar_colors)
        ax.axhline(0, color="#999999", linewidth=0.8)
        ax.set_ylim(-max_abs, max_abs)

        if ratio == 0.0:
            label = "Pure Ridge"
        elif ratio == 1.0:
            label = "Pure Lasso"
        else:
            label = "ElasticNet"

        ax.set_title(f"l1_ratio={ratio}\n({label})", fontsize=10)
        ax.tick_params(axis='x', labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)

        n_zero = int(np.sum(np.abs(coefs) < 1e-10))
        if n_zero > 0:
            ax.text(0.98, 0.98, f"{n_zero} zeroed",
                    transform=ax.transAxes, ha='right', va='top',
                    fontsize=9, color="#e05252")

    axes[0].set_ylabel("Coefficient value")
    plt.suptitle(f"L1/L2 balance  (alpha={self.alpha})", fontsize=12)
    plt.tight_layout()

    print_viz_context(
        diagram="ElasticNet L1/L2 Balance Panel (l1_ratio sweep)",
        solves="Shows how the behavior transitions from pure Ridge to pure Lasso as l1_ratio changes.",
        theory=(
            "ElasticNet combines L1 (Lasso) and L2 (Ridge) penalties with one blending parameter."
            " l1_ratio=0 is pure Ridge: all weights shrink, none zero."
            " l1_ratio=1 is pure Lasso: some weights hit exactly zero."
            " Values between give you the best of both: sparsity + stability for correlated features."
        ),
        formula=(
            "Loss = MSE + alpha * [l1_ratio * sum(|w|) + (1-l1_ratio) * sum(w^2)]\n"
            "l1_ratio=0: Loss = MSE + alpha * sum(w^2)  (pure Ridge)\n"
            "l1_ratio=1: Loss = MSE + alpha * sum(|w|)  (pure Lasso)"
        ),
        constraints=(
            "- l1_ratio must be in [0, 1]\n"
            "- alpha controls overall regularization strength (same for all subplots)\n"
            "- Features must be normalized for fair comparison of bar heights"
        ),
        reading=(
            "- Each subplot = a different l1_ratio value.\n"
            "- Purple bar = active feature (nonzero coefficient).\n"
            "- Grey bar = eliminated feature (coefficient = 0, Lasso effect).\n"
            "- Bar height = coefficient magnitude. Shorter = more regularized."
        ),
    )
    if display:
        return self.show(fig=fig)
    return fig
