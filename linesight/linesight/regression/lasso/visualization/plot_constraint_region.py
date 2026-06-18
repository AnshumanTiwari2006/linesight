import numpy as np
import matplotlib.pyplot as plt
from linesight.utils.validators import _validate_Xy
from linesight.exceptions import LineSightShapeError
from linesight.utils.viz_context import print_viz_context


def plot_constraint_region(self, X, y, display: bool = True):
    """
    For 2-feature models: L2 circle + loss contours (fully vectorized).
    Shows geometrically WHY Ridge cannot zero coefficients (circle never touches axes).
    """
    self._check_fitted("plot_constraint_region")
    X, y = _validate_Xy(X, y)

    if X.shape[1] != 2:
        raise LineSightShapeError(
            f"plot_constraint_region requires exactly 2 features. Got {X.shape[1]}."
        )

    # OLS closed-form reference solution
    X_aug = np.column_stack([np.ones(X.shape[0]), X])
    weights_ols, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
    ols_w = weights_ols[1:]

    ridge_w = self.weights
    radius = float(np.sqrt(np.sum(ridge_w ** 2)))

    span = max(float(np.abs(ols_w).max()) * 1.6, radius * 1.6, 0.5)
    t1 = np.linspace(-span, span, 120)
    t2 = np.linspace(-span, span, 120)
    W1, W2 = np.meshgrid(t1, t2)           # (120, 120)

    # Vectorized MSE: broadcast X @ [W1, W2] using einsum
    # y_pred[i,j,k] = X[k,0]*W1[i,j] + X[k,1]*W2[i,j] + bias
    Y_pred = (
        W1[:, :, np.newaxis] * X[:, 0]
        + W2[:, :, np.newaxis] * X[:, 1]
        + self.bias
    )  # shape (120, 120, n)
    Z = np.mean((Y_pred - y) ** 2, axis=2)   # shape (120, 120)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.contour(W1, W2, Z, levels=16, cmap="Blues", alpha=0.65)

    # L1 diamond — corners sit ON the axes -> forces one weight to 0
    lasso_radius = float(np.sum(np.abs(self.weights)))
    diamond_x = [lasso_radius, 0, -lasso_radius, 0, lasso_radius]
    diamond_y = [0, lasso_radius, 0, -lasso_radius, 0]
    ax.plot(diamond_x, diamond_y,
            color="#e05252", linewidth=2, label=f"L1 diamond (r={round(lasso_radius, 3)})")

    ax.scatter(*ols_w, color="navy",   s=80, zorder=5, label="OLS solution")
    ax.scatter(self.weights[0], self.weights[1], color="#e05252", s=80, zorder=5, label="Lasso solution")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel(f"weight[0]  ({self.feature_names_in_[0]})")
    ax.set_ylabel(f"weight[1]  ({self.feature_names_in_[1]})")
    ax.set_title("Lasso: L1 diamond — corners hit axes -> exact zeros (sparsity)")
    ax.legend()
    ax.set_aspect("equal")
    plt.tight_layout()
    print_viz_context(
        diagram="Lasso L1 Constraint Region (Diamond + Contours)",
        solves="Proves GEOMETRICALLY why Lasso creates EXACT zeros (sparsity), unlike Ridge.",
        theory=(
            "Lasso minimizes MSE subject to the constraint sum(|w_j|) <= t."
            " This forms a DIAMOND (L1 ball) in weight space with sharp corners ON the axes."
            " The MSE contours (ellipses) grow outward from the OLS solution."
            " As the contour expands, it hits the L1 diamond at its CORNER,"
            " which lies exactly on a coordinate axis — meaning one weight = exactly 0."
            " This is why Lasso performs automatic feature selection."
        ),
        formula=(
            "Lasso constraint: |w1| + |w2| <= t   (L1 ball, a diamond)\n"
            "MSE contours: MSE(w) = const  (ellipses centered on OLS)\n"
            "Lasso solution = where smallest contour ellipse touches the L1 diamond"
        ),
        constraints=(
            "- Requires exactly 2 features (2D visualization)\n"
            "- The diamond size = sum(|w_lasso|) from the fitted model\n"
            "- In higher dimensions, the L1 ball has many corners (one per feature)"
        ),
        reading=(
            "- Blue ellipses = MSE contours (inner ellipse = lower loss).\n"
            "- Red diamond = L1 constraint ball (all allowed weight combinations).\n"
            "- Blue dot = OLS solution (unconstrained).\n"
            "- Red dot = Lasso solution (best MSE within the diamond).\n"
            "- Notice: the red dot lands ON an axis corner -> one weight = exactly 0."
        ),
    )
    return self.show(fig=fig) if display else fig
