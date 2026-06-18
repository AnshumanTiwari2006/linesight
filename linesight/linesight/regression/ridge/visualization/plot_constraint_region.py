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

    # L2 circle
    theta_c = np.linspace(0, 2 * np.pi, 300)
    ax.plot(radius * np.cos(theta_c), radius * np.sin(theta_c),
            color="#e05252", linewidth=2, label=f"L2 ball (r={round(radius, 3)})")

    ax.scatter(*ols_w, color="navy",   s=80, zorder=5, label="OLS solution")
    ax.scatter(*ridge_w, color="#e05252", s=80, zorder=5, label="Ridge solution")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel(f"weight[0]  ({self.feature_names_in_[0]})")
    ax.set_ylabel(f"weight[1]  ({self.feature_names_in_[1]})")
    ax.set_title("Ridge: L2 circle — cannot reach axes -> no sparsity")
    ax.legend()
    ax.set_aspect("equal")
    plt.tight_layout()
    print_viz_context(
        diagram="Ridge L2 Constraint Region (Circle + Contours)",
        solves="Proves GEOMETRICALLY why Ridge shrinks coefficients but can never make them exactly zero.",
        theory=(
            "Ridge minimizes MSE subject to the constraint  sum(w_j^2) <= t."
            " This constraint defines a CIRCLE in weight space."
            " The MSE contours are ellipses centered on the OLS solution."
            " The Ridge solution is where the smallest contour ellipse TOUCHES the circle."
            " Since a circle has no corners, it touches the ellipse on its curved side"
            " — never at a coordinate axis — so no weight becomes exactly 0."
        ),
        formula=(
            "Ridge constraint: w1^2 + w2^2 <= t   (L2 ball, a circle)\n"
            "MSE contours: MSE(w) = const  (ellipses centered on OLS)\n"
            "Ridge solution = where smallest contour ellipse touches the L2 circle"
        ),
        constraints=(
            "- Requires exactly 2 features (2D weight space visualization)\n"
            "- The L2 ball radius r = sqrt(sum(w_ridge^2)) from the fitted model\n"
            "- The geometric argument holds for any dimension (only visualizable in 2D)"
        ),
        reading=(
            "- Blue ellipses = MSE contours (inner = lower loss).\n"
            "- Red circle = L2 constraint ball (all allowed weight combinations).\n"
            "- Blue dot = OLS solution (no constraint, minimum MSE).\n"
            "- Red dot = Ridge solution (best MSE WITHIN the circle).\n"
            "- Notice: the red dot never lands on the axes -> no exact zeros."
        ),
    )
    return self.show(fig=fig) if display else fig
