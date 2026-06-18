import numpy as np
import matplotlib.pyplot as plt
import warnings
from linesight.utils.validators import _validate_Xy
from linesight.utils.viz_context import print_viz_context


def plot_feature_elimination(self, X, y, alphas=None, feature_names=None, display: bool = True):
    """
    Show which features get eliminated (zeroed) as alpha increases.
    Annotates the alpha at which each feature hits zero.
    """
    self._check_fitted("plot_feature_elimination")
    X, y = _validate_Xy(X, y)

    if alphas is None:
        alphas = np.logspace(-3, 1, 60)

    from linesight.regression.lasso.core import LassoRegression
    names = feature_names or self.feature_names_in_
    n_features = X.shape[1]
    coef_paths = np.zeros((len(alphas), n_features))

    for i, a in enumerate(alphas):
        tmp = LassoRegression(alpha=a, epochs=self.epochs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tmp.fit(X, y)
        coef_paths[i] = tmp.weights

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, n_features))

    for j, (name, color) in enumerate(zip(names, colors)):
        path = coef_paths[:, j]
        ax.plot(alphas, path, color=color, linewidth=1.5, label=name)

        # Find first alpha where this coefficient hits 0
        zero_indices = np.where(path == 0)[0]
        if len(zero_indices) > 0:
            elim_idx = zero_indices[0]
            elim_alpha = alphas[elim_idx]
            ax.axvline(elim_alpha, color=color, linestyle=":", linewidth=0.8, alpha=0.5)
            ax.annotate(f"{name}\nelim. alpha={round(elim_alpha, 3)}",
                       xy=(elim_alpha, 0),
                       xytext=(elim_alpha * 1.1, max(path) * 0.3),
                       fontsize=7, color=color)

    ax.axvline(self.alpha, color="#333333", linestyle="--", linewidth=1,
               label=f"Current alpha={self.alpha}")
    ax.set_xscale("log")
    ax.set_xlabel("alpha (log scale)")
    ax.set_ylabel("Coefficient value")
    ax.set_title("Lasso feature elimination path")
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    print_viz_context(
        diagram="Lasso Feature Elimination Path (Coefficient Lines)",
        solves="Shows the exact alpha threshold at which each feature's coefficient hits exactly zero.",
        theory=(
            "This is the Lasso regularization path. Unlike Ridge where coefficients"
            " smoothly approach zero asymptotically, Lasso coefficients hit EXACTLY zero"
            " at a specific alpha threshold via coordinate descent soft-thresholding."
            " The elimination order depends on each feature's correlation with the target"
            " after removing the other features' contributions."
        ),
        formula=(
            "Lasso: Loss = MSE + alpha * sum(|w_j|)\n"
            "Coordinate descent update: w_j = soft_threshold(rho_j, alpha) / x_j_sq\n"
            "soft_threshold(z, t) = sign(z) * max(|z| - t, 0)"
        ),
        constraints=(
            "- Feature scales affect which feature is eliminated first\n"
            "- For fair comparison across features, use normalize=True\n"
            "- With highly correlated features, Lasso arbitrarily picks one to keep"
        ),
        reading=(
            "- Each colored line = one feature's coefficient across alpha values.\n"
            "- Where a line hits y=0: that feature is eliminated at that alpha.\n"
            "- Vertical dotted lines mark each feature's elimination alpha.\n"
            "- Black dashed line = your current alpha."
        ),
    )
    if display:
        return self.show(fig=fig)
    return fig
