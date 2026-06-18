import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import warnings
from linesight.utils.validators import _validate_Xy
from linesight.exceptions import LineSightNotFittedError


def animate_boundary(self, X, y, step: int = 10, display: bool = True):
    """
    Animate decision boundary movement over training epochs.

    Parameters
    ----------
    step    : int, default 10  — record every step-th epoch (controls memory)
    display : bool, default True
    """
    self._check_fitted("animate_boundary")

    if self._history.is_empty():
        raise LineSightNotFittedError(
            "No training history found. Re-fit with store_history=True to enable animations."
        )

    # Validate FIRST, then check shape
    X, y = _validate_Xy(X, y)
    if X.shape[1] != 2:
        raise ValueError(
            f"animate_boundary requires exactly 2 features. Got {X.shape[1]}."
        )

    from linesight.utils.environment import _detect_environment
    env = _detect_environment()

    history_weights = self._history.weights[::step]
    history_biases  = self._history.biases[::step]

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    x1_vals = np.array([x_min, x_max])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm",
               edgecolors="k", s=50, zorder=3)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(self.feature_names_in_[0])
    ax.set_ylabel(self.feature_names_in_[1])
    line, = ax.plot([], [], "k-", lw=2)
    title = ax.set_title("")
    plt.close(fig)   # prevent double-render in notebooks

    def init():
        line.set_data([], [])
        title.set_text("")
        return line, title

    def update(frame):
        w = history_weights[frame]
        b = history_biases[frame]
        # Boundary: b + w[0]*x1 + w[1]*x2 = 0  =>  x2 = -(b + w[0]*x1) / w[1]
        if abs(w[1]) > 1e-10:
            x2_vals = -(b + w[0] * x1_vals) / w[1]
            line.set_data(x1_vals, x2_vals)
        else:
            line.set_data([], [])
        title.set_text(f"Epoch {frame * step}")
        
        if frame == len(history_weights) - 1 and not self._history.converged:
            ax.text(0.5, 0.5, "WARNING: Loss still decreasing!\nIncrease epochs or learning rate.", 
                    transform=ax.transAxes, ha="center", va="center", 
                    fontsize=12, color="red", alpha=0.8, weight="bold")
            
        return line, title

    ani = animation.FuncAnimation(
        fig, update, frames=len(history_weights),
        init_func=init, blit=True, interval=60
    )

    if display:
        if env in ("jupyter", "colab"):
            from IPython.display import HTML
            return HTML(ani.to_jshtml())
        else:
            plt.show(block=False)
    return ani
