import numpy as np
import matplotlib.pyplot as plt
from linesight.utils.validators import _validate_Xy
from linesight.exceptions import LineSightShapeError
from linesight.utils.array_utils import _add_bias_column


def plot_3d_loss_slice(self, X, y,
                       theta_i: int = 0,
                       theta_j: int = 1,
                       grid_points: int = 40,
                       display: bool = True):
    """
    Show a 2D slice of the loss surface by varying two parameters.

    Holds all theta values at their fitted values EXCEPT theta_i and theta_j,
    which are varied over a grid. Computes loss at each (theta_i, theta_j)
    combination and renders as a 3D surface.

    This is the correct approach for multi-feature models where the full
    loss surface exists in p+1 dimensions and cannot be visualized directly.

    Parameters
    ----------
    X : array-like, shape (n, p)
    y : array-like, shape (n,)
    theta_i : int, default 0
        Index of first parameter to vary (0 = intercept, 1 = first feature, etc.)
    theta_j : int, default 1
        Index of second parameter to vary.
    grid_points : int, default 40
        Resolution of the surface grid.
    display : bool, default True

    Raises
    ------
    LineSightShapeError
        If theta_i == theta_j:
        "theta_i and theta_j must be different indices.
        Received theta_i={i}, theta_j={j}.
        To slice the loss surface you need two DIFFERENT parameters to vary."

    LineSightShapeError
        If theta_i or theta_j is out of range:
        "theta_i={i} is out of range. Your model has {p} features,
        so valid theta indices are 0 (intercept) through {p} (last feature)."
    """
    self._check_fitted("plot_3d_loss_slice")
    X, y = _validate_Xy(X, y)
    X_bias = _add_bias_column(X)
    p = X_bias.shape[1]  # p+1 parameters total (including intercept)

    if theta_i == theta_j:
        raise LineSightShapeError(
            f"theta_i and theta_j must be different indices.\n"
            f"Received theta_i={theta_i}, theta_j={theta_j}.\n"
            f"Choose two different parameters to slice the loss surface."
        )

    for idx, name in [(theta_i, "theta_i"), (theta_j, "theta_j")]:
        if not (0 <= idx < p):
            raise LineSightShapeError(
                f"{name}={idx} is out of range.\n"
                f"Your model has {X.shape[1]} features, so valid theta indices "
                f"are 0 (intercept) through {p - 1} (feature {p - 2}).\n"
                f"Received {name}={idx}."
            )

    theta_fit = np.concatenate(([self.bias], self.weights))
    names = ["intercept"] + getattr(self, 'feature_names_in_',
                                    [f"x{k}" for k in range(X.shape[1])])

    span_i = max(abs(theta_fit[theta_i]) * 3, 0.5)
    span_j = max(abs(theta_fit[theta_j]) * 3, 0.5)

    vals_i = np.linspace(theta_fit[theta_i] - span_i,
                         theta_fit[theta_i] + span_i, grid_points)
    vals_j = np.linspace(theta_fit[theta_j] - span_j,
                         theta_fit[theta_j] + span_j, grid_points)

    Vi, Vj = np.meshgrid(vals_i, vals_j)
    Z = np.zeros_like(Vi)

    for ri in range(grid_points):
        for ci in range(grid_points):
            theta_tmp = theta_fit.copy()
            theta_tmp[theta_i] = Vi[ri, ci]
            theta_tmp[theta_j] = Vj[ri, ci]
            residuals = X_bias @ theta_tmp - y
            Z[ri, ci] = float(np.mean(residuals ** 2))

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Vi, Vj, Z, cmap='Blues', alpha=0.65, linewidth=0)

    # Mark the fitted point
    z_fit = float(np.mean((X_bias @ theta_fit - y) ** 2))
    ax.scatter([theta_fit[theta_i]], [theta_fit[theta_j]], [z_fit],
               color='red', s=80, zorder=5,
               label=f"Fitted: θ{theta_i}={round(theta_fit[theta_i], 3)}, "
                     f"θ{theta_j}={round(theta_fit[theta_j], 3)}")

    ax.set_xlabel(f"θ{theta_i} ({names[theta_i]})")
    ax.set_ylabel(f"θ{theta_j} ({names[theta_j]})")
    ax.set_zlabel("Loss (MSE)")
    ax.set_title(f"Loss surface slice: varying θ{theta_i} ({names[theta_i]}) "
                 f"and θ{theta_j} ({names[theta_j]})\n"
                 f"All other parameters held at fitted values")
    ax.legend()

    if display:
        return self.show(fig=fig)
    return fig
