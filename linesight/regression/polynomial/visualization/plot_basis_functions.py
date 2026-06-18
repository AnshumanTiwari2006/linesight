import numpy as np
import matplotlib.pyplot as plt
from linesight.utils.validators import _validate_Xy
from linesight.utils.colors import FIT_LINE, DATA_POINTS
from linesight.utils.viz_context import print_viz_context


def plot_basis_functions(self, X, y, display: bool = True):
    """
    Decompose the polynomial fit into its individual basis function components.

    What is drawn
    -------------
    Top panel: Each scaled basis function as a separate colored line
        - "a₁·x¹" as one curve
        - "a₂·x²" as another
        - etc.
        - The intercept b shown as a horizontal dashed line
        - The SUM (final fit) shown as a thick blue line

    Bottom panel: The data scatter with final fit overlaid

    Parameters
    ----------
    X : array-like, shape (n, 1)
    y : array-like, shape (n,)
    display : bool, default True

    Raises
    ------
    LineSightShapeError
        If model has degree > 8:
        "plot_basis_functions() supports up to degree 8 for readability.
        Your model has degree {d}. The plot would have {d} overlapping curves."
    """
    from linesight.exceptions import LineSightShapeError
    from linesight.regression.polynomial.engine.expand_features import _expand_features

    self._check_fitted("plot_basis_functions")

    if self.degree > 8:
        raise LineSightShapeError(
            f"plot_basis_functions() supports up to degree 8 for readability.\n"
            f"Your model has degree {self.degree}.\n"
            f"The plot would have {self.degree} overlapping curves."
        )

    X, y = _validate_Xy(X, y)
    x = X.ravel()
    x_range = np.linspace(x.min(), x.max(), 300)
    X_range = x_range.reshape(-1, 1)

    # Build expanded features for the range
    X_poly_range = _expand_features(X_range, self.degree)

    intercept = float(self.theta_[0])
    coefs = self.theta_[1:]

    # Colors for each basis function
    basis_colors = [
        "#1a6fcc", "#2a9d4e", "#e05252", "#7047c4",
        "#e08c2a", "#d9534a", "#5280e0", "#888888"
    ]

    superscripts = {1: "¹", 2: "²", 3: "³", 4: "⁴",
                    5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸"}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8))

    # Intercept horizontal line
    ax1.axhline(intercept, color="#999999", linewidth=1.5,
                linestyle="--", label=f"intercept = {round(intercept, 3)}", alpha=0.8)

    total = np.full(300, intercept)

    for p_idx in range(self.degree):
        degree = p_idx + 1
        coef = float(coefs[p_idx])
        component = coef * X_poly_range[:, p_idx]
        total = total + component
        sup = superscripts.get(degree, f"^{degree}")
        color = basis_colors[p_idx % len(basis_colors)]
        label = f"{round(coef, 3)}·x{sup}"
        ax1.plot(x_range, component, color=color, linewidth=1.5,
                 linestyle="--", alpha=0.75, label=label)

    # Total (final fit) as thick line
    ax1.plot(x_range, total, color=FIT_LINE, linewidth=2.5,
             label="Sum = final fit", zorder=5)
    ax1.axhline(0, color="#cccccc", linewidth=0.6)

    ax1.set_title(f"Polynomial degree {self.degree}: basis function decomposition")
    ax1.set_ylabel("Scaled component value")
    ax1.legend(loc="best", fontsize=8, ncol=2)
    ax1.spines[["top", "right"]].set_visible(False)

    # Bottom: data + fit
    ax2.scatter(x, y, color=DATA_POINTS, edgecolors="white",
                s=55, linewidths=0.8, zorder=3)
    ax2.plot(x_range, total, color=FIT_LINE, linewidth=2.5, label="Fit")
    ax2.set_xlabel("X")
    ax2.set_ylabel("y")
    ax2.set_title("Final fit on data")
    ax2.legend()
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()

    print_viz_context(
        diagram="Polynomial Basis Function Decomposition",
        solves="Reveals how each individual power of x contributes to the final curved fit.",
        theory=(
            "Polynomial regression transforms the single feature x into multiple features:"
            " [x, x^2, x^3, ..., x^d] (feature expansion).  This turns a non-linear problem"
            " into a multi-feature LINEAR problem that standard gradient descent can solve."
            " Each new feature x^k gets its own coefficient w_k, and the final prediction is"
            " their weighted sum.  The top panel deconstructs this sum."
        ),
        formula=(
            "y = b + w1*x^1 + w2*x^2 + ... + wd*x^d\n"
            "Feature expansion: phi(x) = [x, x^2, x^3, ..., x^d]\n"
            "Each colored line = w_k * x^k  (one term of the sum)"
        ),
        constraints=(
            "- Only works for single-input polynomial models (x must be 1D)\n"
            "- Supported up to degree 8 (higher degrees produce unreadable plots)\n"
            "- Very high degrees cause numerical instability; always normalize before fitting"
        ),
        reading=(
            "- Top panel: each dashed colored line = one basis function (one power of x).\n"
            "- Horizontal dashed grey line = intercept (b).\n"
            "- Thick blue line = the SUM of all components = the final prediction.\n"
            "- Bottom panel: the final fit overlaid on your actual data."
        ),
    )
    if display:
        return self.show(fig=fig)
    return fig
