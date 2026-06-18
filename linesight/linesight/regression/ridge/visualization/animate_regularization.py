import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from linesight.utils.colors import DATA_POINTS, FIT_LINE, COLOR_RIDGE
from linesight.utils.validators import _validate_Xy


def animate_regularization(self, X, y, alphas=None,
                            interval: int = 300, display: bool = True):
    """
    Animate how the Ridge regression line changes as alpha increases.

    Only works for single-feature models (so the line can be shown directly).
    For multi-feature models, use plot_coefficient_shrinkage() instead.

    What is drawn
    -------------
    Each frame fits a Ridge model with a different alpha and shows:
    - The regression line for that alpha
    - Current coefficient values in the corner
    - Title: "Alpha = 0.001 → line is flexible"
             "Alpha = 100.0 → line is nearly flat (over-regularized)"

    What this teaches
    -----------------
    Students watch the line physically stiffen as alpha increases.
    At very high alpha, the line approaches the horizontal mean line,
    because the model has been forced to make all coefficients near zero.

    Parameters
    ----------
    X : array-like, shape (n, 1) — single feature required
    y : array-like, shape (n,)
    alphas : list of float, optional
        Default: 30 values log-spaced from 1e-4 to 1e3
    interval : int, default 300
    display : bool, default True
    """
    from linesight.utils.validators import _validate_Xy
    from linesight.exceptions import LineSightShapeError
    from linesight.regression.ridge.core import RidgeRegression

    X, y = _validate_Xy(X, y)

    if X.shape[1] != 1:
        raise LineSightShapeError(
            "animate_regularization() requires single-feature X.\n"
            "For multi-feature models, use plot_coefficient_shrinkage()."
        )

    if alphas is None:
        alphas = np.logspace(-4, 3, 30)

    x = X.ravel()
    x_line = np.linspace(x.min(), x.max(), 200)

    # Pre-compute all models
    lines_y = []
    coef_vals = []
    for alpha in alphas:
        tmp = RidgeRegression(alpha=alpha,
                              learning_rate=self.learning_rate,
                              epochs=self.epochs)
        tmp.fit(X, y)
        lines_y.append(tmp.predict(x_line.reshape(-1, 1)))
        coef_vals.append(round(float(tmp.weights[0]), 4))

    y_all = np.concatenate(lines_y)
    y_margin = (y.max() - y.min()) * 0.2

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, color=DATA_POINTS, edgecolors="white",
               linewidths=0.8, s=60, zorder=3)
    line, = ax.plot([], [], color=FIT_LINE, linewidth=2)
    title = ax.set_title("")
    coef_text = ax.text(0.02, 0.95, "", transform=ax.transAxes,
                        fontsize=9, verticalalignment='top')

    ax.set_xlim(x.min() - 0.5, x.max() + 0.5)
    ax.set_ylim(y.min() - y_margin, y.max() + y_margin)
    ax.set_xlabel("X")
    ax.set_ylabel("ŷ")
    ax.spines[["top", "right"]].set_visible(False)

    def _update(frame):
        line.set_data(x_line, lines_y[frame])
        alpha = alphas[frame]
        coef = coef_vals[frame]
        title.set_text(f"Alpha = {round(alpha, 4)}  —  slope = {coef}")
        coef_text.set_text(f"As alpha increases,\nthe slope shrinks toward 0")
        return line, title, coef_text

    anim = animation.FuncAnimation(
        fig, _update, frames=len(alphas), interval=interval, blit=False
    )
    plt.tight_layout()

    if display:
        return self.show(animation_obj=anim)
    return anim
