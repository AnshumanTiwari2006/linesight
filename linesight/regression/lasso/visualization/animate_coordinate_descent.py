import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from linesight.utils.colors import DATA_POINTS, FIT_LINE, RESIDUAL_POS
from linesight.exceptions import LineSightShapeError


def animate_coordinate_descent(self, X, y, interval: int = 100, display: bool = True):
    """
    Animate Lasso's coordinate descent: watch ONE COEFFICIENT AT A TIME update.

    This is unique to Lasso/ElasticNet. Unlike gradient descent which updates
    ALL coefficients simultaneously each step, coordinate descent updates them
    one by one in sequence.

    What is drawn (for 2-feature models)
    --------------------------------------
    - Left subplot: scatter of X[:,0] vs y with current fit line for feature 0
    - Right subplot: scatter of X[:,1] vs y with current fit line for feature 1
    - The currently-updating feature is highlighted (other grayed out)
    - A bar chart at the bottom shows current coefficient magnitudes
    - Title: "Epoch 12, updating feature 1 (x₁)"

    For >2 features:
    ----------------
    - Single plot: bar chart of all coefficient magnitudes, updating each epoch
    - One bar lights up per update step showing which coordinate is being updated

    What this teaches
    -----------------
    The sequential nature of coordinate descent. Students see that only one
    coefficient changes per update, and that the coefficient can jump to exactly
    zero (soft-thresholding). This is impossible to understand from text alone.

    Parameters
    ----------
    X : array-like, shape (n, p)
    y : array-like, shape (n,)
    interval : int, default 100
    display : bool, default True

    Requires store_history=True (history stores coefficients per epoch).
    """
    self._check_fitted("animate_coordinate_descent")

    import warnings
    from linesight.exceptions import LineSightDataWarning

    if self._history.is_empty():
        warnings.warn(
            "No history. Re-fit with store_history=True.",
            LineSightDataWarning, stacklevel=2
        )
        return None

    from linesight.utils.validators import _validate_Xy
    X, y = _validate_Xy(X, y)
    n, p = X.shape

    weights = self._history.weights

    feature_names = getattr(self, 'feature_names_in_',
                            [f"x{j}" for j in range(p)])

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(feature_names, np.abs(weights[0]),
                  color=[FIT_LINE] * p)
    ax.axhline(0, color="#cccccc", linewidth=0.8)
    title = ax.set_title("")
    ax.set_ylabel("|coefficient|")
    ax.set_xlabel("Feature")
    ax.spines[["top", "right"]].set_visible(False)

    def _update(frame):
        coefs = np.abs(weights[frame])
        epoch = frame + 1

        for bar, height in zip(bars, coefs):
            bar.set_height(height)

        # Highlight which feature would be updated next
        # In coordinate descent, feature updated = epoch mod n_features
        active_j = frame % p
        for j, bar in enumerate(bars):
            bar.set_color(RESIDUAL_POS if j == active_j else FIT_LINE)
            bar.set_alpha(1.0 if j == active_j else 0.5)

        # Count zeroed coefficients
        n_zeroed = int(np.sum(coefs < 1e-10))
        title.set_text(
            f"Epoch {epoch} — updating '{feature_names[active_j]}' — "
            f"{n_zeroed}/{p} features zeroed"
        )
        return bars

    anim = animation.FuncAnimation(
        fig, _update, frames=len(weights),
        interval=interval, blit=False
    )
    plt.tight_layout()

    if display:
        return self.show(animation_obj=anim)
    return anim
