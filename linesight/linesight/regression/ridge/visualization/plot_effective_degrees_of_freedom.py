import numpy as np
import matplotlib.pyplot as plt
from linesight.utils.validators import _validate_Xy
from linesight.utils.array_utils import _add_bias_column
from linesight.utils.colors import FIT_LINE, RESIDUAL_POS


def plot_effective_degrees_of_freedom(self, X, y,
                                       alphas=None,
                                       display: bool = True):
    """
    Plot the model's effective degrees of freedom as alpha varies.

    Formula
    -------
    EDF(α) = trace( X(XᵀX + αI)⁻¹Xᵀ )

    This uses the normal equation form (not gradient descent), which gives
    the exact analytical answer. The normal equation form of Ridge is:
        theta = (XᵀX + αI)⁻¹ Xᵀy

    The hat matrix H = X(XᵀX + αI)⁻¹Xᵀ maps y to fitted values.
    Its trace = sum of its eigenvalues = effective degrees of freedom.

    What is drawn
    -------------
    X-axis: alpha (log scale)
    Y-axis: EDF
    Horizontal dashed line at p (unregularized model's EDF)
    Horizontal dashed line at 1 (intercept-only model's EDF)
    Vertical line at self.alpha (the current model's alpha)

    Parameters
    ----------
    X : array-like, shape (n, p)
    y : array-like, shape (n,)
    alphas : list of float, optional
        Default: 50 log-spaced values from 1e-4 to 1e4
    display : bool, default True

    Warns
    -----
    LineSightDataWarning if n < p:
        "n={n} < p={p}: the system is underdetermined. EDF values may be
        unreliable. Regularization is especially important here."
    """
    import warnings
    from linesight.exceptions import LineSightDataWarning

    self._check_fitted("plot_effective_degrees_of_freedom")
    X, y = _validate_Xy(X, y)
    X_bias = _add_bias_column(X)
    n, q = X_bias.shape  # q = p+1 (includes intercept)

    if n < q:
        warnings.warn(
            f"n={n} < p={q - 1}: the system is underdetermined.\n"
            f"EDF values may be unreliable. Ridge regularization is critical here.",
            LineSightDataWarning, stacklevel=2
        )

    if alphas is None:
        alphas = np.logspace(-4, 4, 80)

    XtX = X_bias.T @ X_bias  # shape (q, q)
    edfs = []

    for alpha in alphas:
        # (XᵀX + αI)⁻¹ — the alpha penalty goes on all params except intercept
        # Strictly: penalty only on feature weights, not intercept.
        # Build penalty matrix: zeros for intercept (index 0), alpha for rest
        penalty = np.zeros((q, q))
        penalty[1:, 1:] = alpha * np.eye(q - 1)

        try:
            inv = np.linalg.inv(XtX + penalty)
            H = X_bias @ inv @ X_bias.T          # hat matrix (n, n)
            edf = float(np.trace(H))
        except np.linalg.LinAlgError:
            edf = float('nan')
        edfs.append(edf)

    edfs = np.array(edfs)
    current_alpha = self.alpha

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogx(alphas, edfs, color=FIT_LINE, linewidth=2,
                label="EDF(α)")
    ax.axhline(q - 1, color="#aaaaaa", linewidth=1.2, linestyle="--",
               label=f"Full model (EDF = {q - 1} features)")
    ax.axhline(1, color="#cccccc", linewidth=1, linestyle=":",
               label="Intercept-only (EDF = 1)")
    ax.axvline(current_alpha, color=RESIDUAL_POS, linewidth=1.5,
               linestyle="--",
               label=f"Current α = {current_alpha}")

    # Annotate current EDF
    current_idx = np.argmin(np.abs(np.array(alphas) - current_alpha))
    if not np.isnan(edfs[current_idx]):
        ax.scatter([current_alpha], [edfs[current_idx]],
                   color=RESIDUAL_POS, s=70, zorder=5)
        ax.annotate(f"  EDF = {round(edfs[current_idx], 2)}",
                    xy=(current_alpha, edfs[current_idx]),
                    fontsize=9, color=RESIDUAL_POS)

    ax.set_xlabel("Alpha (regularization strength) — log scale")
    ax.set_ylabel("Effective degrees of freedom")
    ax.set_title(
        "How regularization reduces model complexity\n"
        "EDF→p at α=0 (full model), EDF→0 at α→∞ (constant model)"
    )
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)

    if display:
        return self.show(fig=fig)
    return fig
