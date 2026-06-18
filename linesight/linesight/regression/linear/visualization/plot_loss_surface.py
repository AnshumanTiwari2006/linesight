import numpy as np
import matplotlib.pyplot as plt
import warnings
from linesight.utils.validators import _validate_Xy
from linesight.exceptions import LineSightDataWarning
from linesight.utils.viz_context import print_viz_context


def plot_loss_surface(self, X, y, m_range=None, b_range=None,
                      grid_points: int = 50, display: bool = True):
    """
    Plot the 3D loss surface over the (slope, intercept) parameter space.

    What is drawn
    -------------
    - 3D surface where Z = MSE loss at each (m, b) combination
    - The bowl shape shows WHY gradient descent works — loss decreases in all
      directions toward the minimum
    - Red dot marks the model's fitted (m, b) on the surface
    - If store_history=True, the optimization path is drawn on the surface

    Parameters
    ----------
    X : array-like, shape (n, 1)
    y : array-like, shape (n,)
    m_range : tuple (min, max), optional
        Range of slope values to plot. Default: fitted m +/- 3 * abs(fitted m)
    b_range : tuple (min, max), optional
        Range of intercept values. Default: fitted b +/- 3 * abs(fitted b) + 1
    grid_points : int, default 50
        Resolution of the surface grid. Higher = smoother but slower.
    display : bool, default True

    Performance warning
    -------------------
    Computes loss for grid_points^2 parameter combinations.
    With grid_points=50 and n=10000 samples, this is 25M multiplications.
    A warning is issued for n > 5000.

    Only works for single-feature models.
    """
    self._check_fitted("plot_loss_surface")
    X, y = _validate_Xy(X, y)

    if X.shape[1] != 1:
        warnings.warn(
            "plot_loss_surface() only works for single-feature models. "
            "For multi-feature models, the loss surface exists in N+1 dimensions "
            "and cannot be shown as a 3D plot.",
            LineSightDataWarning, stacklevel=2
        )
        return None

    if X.shape[0] > 5000:
        warnings.warn(
            f"plot_loss_surface() on {X.shape[0]} samples may be slow. "
            f"Consider passing a subset: plot_loss_surface(X[:2000], y[:2000])",
            LineSightDataWarning, stacklevel=2
        )

    x = X.ravel()
    m_fit = float(self.m)
    b_fit = float(self.b)

    if m_range is None:
        span_m = max(abs(m_fit) * 3, 1.0)
        m_range = (m_fit - span_m, m_fit + span_m)
    if b_range is None:
        span_b = max(abs(b_fit) * 3, 1.0)
        b_range = (b_fit - span_b, b_fit + span_b)

    m_vals = np.linspace(m_range[0], m_range[1], grid_points)
    b_vals = np.linspace(b_range[0], b_range[1], grid_points)
    M, B = np.meshgrid(m_vals, b_vals)

    # Vectorized loss computation over the grid
    # x shape: (n,), M shape: (gp, gp)
    # Expand dims to broadcast: (gp, gp, n)
    x_3d = x[np.newaxis, np.newaxis, :]        # (1, 1, n)
    M_3d = M[:, :, np.newaxis]                  # (gp, gp, 1)
    B_3d = B[:, :, np.newaxis]                  # (gp, gp, 1)
    y_3d = y[np.newaxis, np.newaxis, :]         # (1, 1, n)

    Z = np.mean((M_3d * x_3d + B_3d - y_3d) ** 2, axis=2)  # (gp, gp)

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(M, B, Z, cmap='Blues', alpha=0.7, linewidth=0)
    ax.scatter([m_fit], [b_fit], [float(np.mean((m_fit * x + b_fit - y)**2))],
               color='red', s=60, zorder=5, label=f"Fitted: m={round(m_fit,3)}, b={round(b_fit,3)}")

    # Plot optimization path if history available
    if not self._history.is_empty():
        ms = self._history.weights
        bs = self._history.biases
        ls = self._history.losses
        ax.plot(ms, bs, ls, color='red', linewidth=1, alpha=0.6, label="Gradient path")

    ax.set_xlabel("Slope (m)")
    ax.set_ylabel("Intercept (b)")
    ax.set_zlabel("Loss (MSE)")
    ax.set_title("Loss surface")
    ax.legend()
    plt.tight_layout()

    print_viz_context(
        diagram="3D Loss Surface (MSE landscape)",
        solves="Reveals the 'bowl' that gradient descent is rolling down to find the optimal m and b.",
        theory=(
            "For every possible combination of slope m and intercept b, the MSE loss"
            " has one specific value. This forms a bowl-shaped surface in 3D space."
            " Gradient descent is the algorithm that starts at a random point on"
            " this bowl and takes small steps downhill (opposite to the gradient)"
            " until it reaches the lowest point: the optimal parameters."
        ),
        formula=(
            "MSE(m, b) = (1/n) * sum((y_i - (m*x_i + b))^2)\n"
            "Gradient step: m := m - lr * dMSE/dm   |   b := b - lr * dMSE/db"
        ),
        constraints=(
            "- Only works for single-feature models (3D = 2 params + 1 loss axis)\n"
            "- The bowl shape assumes a CONVEX loss (linear regression IS convex)\n"
            "- The shown path only plots if store_history=True during fit()"
        ),
        reading=(
            "- The 3D surface is the MSE bowl. Every point on it is a possible (m, b) pair.\n"
            "- The colored dot/path is the route gradient descent actually took.\n"
            "- A path spiraling to the center = convergence.\n"
            "- A path bouncing wildly = learning rate is too high."
        ),
    )
    if display:
        return self.show(fig=fig)
    return fig
