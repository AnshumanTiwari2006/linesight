import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from linesight.utils.validators import _validate_Xy
from linesight.utils.colors import DATA_POINTS, FIT_LINE


def animate_degree_increase(self, X, y, max_degree: int = 9,
                             interval: int = 800, display: bool = True):
    """
    Animate the polynomial curve changing as degree increases from 1 to max_degree.

    What this teaches
    -----------------
    Students watch the curve go from a straight line (degree 1) through
    increasing flexibility, and eventually start fitting noise (overfitting).
    This is the single clearest possible demonstration of the bias-variance
    tradeoff: low degree = high bias (underfit), high degree = high variance (overfit).

    Each frame shows one degree. The curve re-fits from scratch for each degree.
    The title shows: "Degree N — R² = 0.923"

    Parameters
    ----------
    X : array-like
    y : array-like
    max_degree : int, default 9
    interval : int, default 800
        Milliseconds per frame. 800ms lets students read the R² before moving on.
    display : bool, default True
    """
    X, y = _validate_Xy(X, y)
    x = X.ravel()
    x_curve = np.linspace(x.min(), x.max(), 300)

    from linesight.regression.polynomial.core import PolynomialRegression
    from linesight.metrics import r2

    # Pre-compute all degrees' curves so animation is smooth
    curves = []
    r2_vals = []
    for deg in range(1, max_degree + 1):
        tmp = PolynomialRegression(
            degree=deg,
            learning_rate=self.learning_rate,
            epochs=self.epochs,
            normalize=True,
        )
        tmp.fit(X, y)
        curves.append(tmp.predict(x_curve.reshape(-1, 1)))
        r2_vals.append(round(r2(y, tmp.predict(X)), 3))

    y_margin = (y.max() - y.min()) * 0.2
    y_min_plot = y.min() - y_margin
    y_max_plot = y.max() + y_margin

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, color=DATA_POINTS, edgecolors="white",
               linewidths=0.8, s=60, zorder=3)
    line, = ax.plot([], [], color=FIT_LINE, linewidth=2.5)
    title = ax.set_title("")

    ax.set_xlim(x.min() - 0.2, x.max() + 0.2)
    ax.set_ylim(y_min_plot, y_max_plot)
    ax.set_xlabel("X")
    ax.set_ylabel("ŷ")
    ax.spines[["top", "right"]].set_visible(False)

    def _update(frame):
        deg = frame + 1
        y_curve = curves[frame]

        # Clip extreme values so overfitting doesn't destroy the plot axis
        y_curve_clipped = np.clip(y_curve, y_min_plot - 5, y_max_plot + 5)

        line.set_data(x_curve, y_curve_clipped)
        r2_v = r2_vals[frame]
        title.set_text(f"Degree {deg} — R² = {r2_v}")
        return line, title

    anim = animation.FuncAnimation(
        fig, _update, frames=max_degree, interval=interval, blit=False
    )
    plt.tight_layout()

    if display:
        return self.show(animation_obj=anim)
    return anim
