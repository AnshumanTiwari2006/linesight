import numpy as np
import matplotlib.pyplot as plt
import warnings
from linesight.utils.validators import _validate_Xy
from linesight.utils.viz_context import print_viz_context


def plot_coefficient_shrinkage(self, X, y, alphas=None, display: bool = True):
    """
    Plot coefficient paths as alpha increases (Ridge regularization path).
    Each line is one feature. All start near OLS solution and shrink toward 0.
    """
    self._check_fitted("plot_coefficient_shrinkage")
    X, y = _validate_Xy(X, y)

    if alphas is None:
        alphas = np.logspace(-3, 3, 50)

    from linesight.regression.ridge.core import RidgeRegression
    n_features = X.shape[1]
    coef_paths = np.zeros((len(alphas), n_features))

    for i, a in enumerate(alphas):
        tmp = RidgeRegression(alpha=a, learning_rate=self.learning_rate,
                              epochs=self.epochs, store_history=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tmp.fit(X, y)
        coef_paths[i] = tmp.weights

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, n_features))

    for j, (name, color) in enumerate(zip(self.feature_names_in_, colors)):
        ax.plot(alphas, coef_paths[:, j], color=color, linewidth=1.5, label=name)

    ax.axvline(self.alpha, color="#e05252", linestyle="--", linewidth=1,
               label=f"Current alpha={self.alpha}")
    ax.set_xscale("log")
    ax.set_xlabel("alpha (log scale)")
    ax.set_ylabel("Coefficient value")
    ax.set_title("Ridge coefficient shrinkage path")
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    print_viz_context(
        diagram="Ridge Coefficient Shrinkage Path",
        solves="Shows how increasing the L2 penalty alpha gradually shrinks all feature weights toward zero.",
        theory=(
            "Ridge regression adds alpha * sum(w_j^2) to the MSE loss."
            " As alpha increases, the optimizer must trade off prediction accuracy"
            " against keeping weights small. The solution has a closed form:"
            " w_ridge = (X'X + alpha*I)^{-1} X'y.  As alpha -> inf, all weights -> 0."
            " Crucially, Ridge NEVER reaches exactly 0 — it only shrinks."
        ),
        formula=(
            "Loss = MSE + alpha * sum(w_j^2)\n"
            "Ridge solution: w = (X'X + alpha*I)^{-1} X'y"
        ),
        constraints=(
            "- alpha must be >= 0 (negative alpha would expand, not shrink)\n"
            "- All features should be on the same scale (normalize=True) for fair shrinkage\n"
            "- Ridge SHRINKS coefficients but never eliminates them (use Lasso for that)"
        ),
        reading=(
            "- Each line is one feature's coefficient as alpha increases (left to right).\n"
            "- At alpha=0 (far left): Ridge solution = OLS solution.\n"
            "- As alpha grows: all lines trend toward 0 but never touch it.\n"
            "- Dashed vertical line = the alpha you actually used in this model."
        ),
    )
    if display:
        return self.show(fig=fig)
    return fig
