import numpy as np
import matplotlib.pyplot as plt
from linesight.utils.colors import (COLOR_LINEAR, COLOR_RIDGE,
                                     COLOR_LASSO, COLOR_ELASTICNET)


def compare_regularization_methods(self, X, y, alpha=None, display: bool = True):
    """
    Fit four models on the same data and compare coefficient magnitudes side by side.

    Models compared: LinearRegression, Ridge, Lasso, ElasticNet (same alpha).

    What is drawn
    -------------
    2×2 subplot grid. Each subplot is a bar chart of coefficient values for
    one model. All four use the same Y-axis scale so magnitudes are comparable.

    The visual tells the story:
    - Linear: largest coefficients (no shrinkage)
    - Ridge: all coefficients smaller, none are zero
    - Lasso: some coefficients exactly zero (sparse)
    - ElasticNet: mix of both behaviors

    Parameters
    ----------
    X : array-like
    y : array-like
    alpha : float, optional
        Regularization strength. Defaults to self.alpha.
    display : bool, default True
    """
    self._check_fitted("compare_regularization_methods")

    from linesight.utils.validators import _validate_Xy
    from linesight.regression.multiple.core import MultipleLinearRegression
    from linesight.regression.ridge.core import RidgeRegression
    from linesight.regression.lasso.core import LassoRegression
    from linesight.regression.elasticnet.core import ElasticNetRegression

    X, y = _validate_Xy(X, y)

    if alpha is None:
        alpha = self.alpha

    feature_names = getattr(self, 'feature_names_in_',
                            [f"x{j}" for j in range(X.shape[1])])

    models = {
        "Linear\n(no regularization)": MultipleLinearRegression(
            learning_rate=0.01, epochs=self.epochs, normalize=True),
        f"Ridge\n(α={alpha})": RidgeRegression(
            alpha=alpha, learning_rate=0.01, epochs=self.epochs),
        f"Lasso\n(α={alpha})": LassoRegression(
            alpha=alpha, epochs=self.epochs),
        f"ElasticNet\n(α={alpha}, l1={self.l1_ratio})": ElasticNetRegression(
            alpha=alpha, l1_ratio=self.l1_ratio,
            epochs=self.epochs),
    }
    colors = [COLOR_LINEAR, COLOR_RIDGE, COLOR_LASSO, COLOR_ELASTICNET]

    # Fit all and collect coefficients
    coefs_dict = {}
    for name, model in models.items():
        model.fit(X, y)
        coefs_dict[name] = model.weights

    # Shared y-axis scale
    all_vals = np.concatenate(list(coefs_dict.values()))
    max_abs = max(abs(all_vals).max() * 1.15, 0.01)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes = axes.ravel()

    for i, (name, coefs) in enumerate(coefs_dict.items()):
        ax = axes[i]
        color = colors[i]
        bar_colors = [color if c != 0.0 else "#cccccc" for c in coefs]

        ax.bar(feature_names, coefs, color=bar_colors)
        ax.axhline(0, color="#999999", linewidth=0.8)
        ax.set_ylim(-max_abs, max_abs)
        ax.set_title(name, fontsize=10)
        ax.set_ylabel("Coefficient")
        ax.tick_params(axis='x', labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)

        # Annotate zeroed count for Lasso and ElasticNet
        n_zero = int(np.sum(np.abs(coefs) < 1e-10))
        if n_zero > 0:
            ax.text(0.98, 0.98, f"{n_zero} zeroed",
                    transform=ax.transAxes, ha='right', va='top',
                    fontsize=9, color="#e05252")

    plt.suptitle(f"Regularization comparison — same data, same alpha={alpha}",
                 fontsize=12)
    plt.tight_layout()

    if display:
        return self.show(fig=fig)
    return fig
