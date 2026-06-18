import numpy as np
import matplotlib.pyplot as plt
from linesight.utils.colors import FIT_LINE, RESIDUAL_POS, COLOR_RIDGE
from linesight.utils.viz_context import print_viz_context


def plot_bias_variance_tradeoff(self, X, y, alphas=None, display: bool = True):
    """
    Show training error and validation error as alpha (regularization) increases.

    What is drawn
    -------------
    - X-axis: alpha (log scale, increasing = more regularization)
    - Y-axis: MSE
    - Blue line: training MSE (always increases as alpha increases, model gets worse on training)
    - Red line: validation MSE (forms a U-shape: too little alpha = overfit,
      too much alpha = underfit, sweet spot in the middle)
    - Vertical dashed line marking the optimal alpha (min validation MSE)

    What this teaches
    -----------------
    This is the bias-variance tradeoff made visible. It directly answers
    "how do I choose alpha?" — pick where validation MSE is lowest.

    Parameters
    ----------
    X : array-like, shape (n, p)
    y : array-like, shape (n,)
    alphas : list of float, optional
        Default: 50 values log-spaced from 1e-4 to 1e2
    display : bool, default True
    """
    self._check_fitted("plot_bias_variance_tradeoff")

    from linesight.utils.validators import _validate_Xy
    from linesight.regression.ridge.core import RidgeRegression
    from linesight.metrics import mse
    import numpy as np

    X, y = _validate_Xy(X, y)
    n = X.shape[0]

    if alphas is None:
        alphas = np.logspace(-4, 2, 50)

    # Simple 80/20 split for validation
    split = int(0.8 * n)
    perm = np.random.permutation(n)
    X_tr, y_tr = X[perm[:split]], y[perm[:split]]
    X_val, y_val = X[perm[split:]], y[perm[split:]]

    train_errors = []
    val_errors = []

    for alpha in alphas:
        tmp = RidgeRegression(
            alpha=alpha,
            learning_rate=self.learning_rate,
            epochs=self.epochs
        )
        tmp.fit(X_tr, y_tr)
        train_errors.append(mse(y_tr, tmp.predict(X_tr)))
        val_errors.append(mse(y_val, tmp.predict(X_val)))

    best_alpha_idx = int(np.argmin(val_errors))
    best_alpha = alphas[best_alpha_idx]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogx(alphas, train_errors, color=FIT_LINE, linewidth=2,
                label="Training MSE")
    ax.semilogx(alphas, val_errors, color=RESIDUAL_POS, linewidth=2,
                linestyle="--", label="Validation MSE")
    ax.axvline(x=best_alpha, color="#2a9d4e", linewidth=1.5,
               linestyle=":", label=f"Best alpha ≈ {round(best_alpha, 4)}")

    ax.set_xlabel("Alpha (regularization strength) — log scale")
    ax.set_ylabel("MSE")
    ax.set_title("Bias-variance tradeoff: choosing the right alpha")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)

    print_viz_context(
        diagram="Bias-Variance Tradeoff (Ridge Alpha Selection)",
        solves="Answers: what is the BEST alpha? Too small = overfit, too large = underfit.",
        theory=(
            "Every regularized model faces the bias-variance tradeoff."
            " With low alpha, the model has high variance (fits noise, fails on new data)."
            " With high alpha, the model has high bias (too constrained, misses the signal)."
            " The validation error forms a U-shape. The minimum of that U is the optimal alpha."
        ),
        formula=(
            "Total Error = Bias^2 + Variance + Irreducible Noise\n"
            "Training MSE: sum((y_train - y_hat)^2) / n_train\n"
            "Validation MSE: sum((y_val - y_hat)^2) / n_val"
        ),
        constraints=(
            "- Uses a random 80/20 train/validation split (results vary between runs)\n"
            "- The optimal alpha here is for THIS data split only\n"
            "- For production: use cross-validation (k-fold) for a stable estimate"
        ),
        reading=(
            "- Blue line (Training MSE) always rises as alpha increases (model constrained more).\n"
            "- Red dashed (Validation MSE) forms a U: sweet spot at the bottom.\n"
            "- Green dotted vertical line = estimated best alpha for this data.\n"
            "- Large gap between train and val at low alpha = overfitting."
        ),
    )
    if display:
        return self.show(fig=fig)
    return fig
