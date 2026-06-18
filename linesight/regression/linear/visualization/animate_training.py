import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import warnings
from linesight.utils.validators import _validate_Xy
from linesight.exceptions import LineSightDataWarning, LineSightShapeError


def animate_training(self, X, y, interval: int = 50,
                     skip_frames: int = 5, display: bool = True):
    """
    Animate gradient descent: watch the regression line converge epoch by epoch.

    Requires store_history=True during fit().

    Parameters
    ----------
    X            : array-like, shape (n, 1)
    y            : array-like, shape (n,)
    interval     : int, default 50   — milliseconds between frames
    skip_frames  : int, default 5    — render every Nth epoch (saves memory)
    display      : bool, default True

    Returns
    -------
    HTML in Jupyter | FuncAnimation object when display=False
    """
    self._check_fitted("animate_training")

    if self._history.is_empty():
        warnings.warn(
            "No history to animate. Re-fit with store_history=True.",
            LineSightDataWarning, stacklevel=2
        )
        return None

    X, y = _validate_Xy(X, y)
    if X.shape[1] != 1:
        raise LineSightShapeError(
            "animate_training() only works for single-feature (LinearRegression) models."
        )

    x = X.ravel()

    # LinearRegression stores m scalars in .weights and b scalars in .biases
    m_history = self._history.weights   # list of floats (slope per epoch)
    b_history = self._history.biases    # list of floats (intercept per epoch)
    losses     = self._history.losses

    # Select frames at skip_frames interval, always include last frame
    frame_indices = list(range(0, len(m_history), skip_frames))
    if frame_indices[-1] != len(m_history) - 1:
        frame_indices.append(len(m_history) - 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, color="#888888", edgecolors="white",
               linewidths=0.8, s=60, zorder=3, label="Data")

    x_line = np.linspace(x.min(), x.max(), 200)
    line,   = ax.plot([], [], color="#1a6fcc", linewidth=2, label="Fit")
    title    = ax.set_title("")
    eq_text  = ax.text(0.02, 0.95, "", transform=ax.transAxes,
                       fontsize=9, verticalalignment="top", color="#555555")

    ax.set_xlim(x.min() - 0.5, x.max() + 0.5)
    y_margin = (y.max() - y.min()) * 0.15
    ax.set_ylim(y.min() - y_margin, y.max() + y_margin)
    ax.set_xlabel(self.feature_names_in_[0] if hasattr(self, "feature_names_in_") else "X")
    ax.set_ylabel("y")
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    plt.close(fig)   # prevent double-render in Jupyter

    def _update(frame_idx):
        idx  = frame_indices[frame_idx]
        m    = m_history[idx]    # slope at this epoch
        b    = b_history[idx]    # intercept at this epoch
        loss = losses[idx] if idx < len(losses) else float("nan")
        epoch = idx + 1

        y_line = m * x_line + b
        line.set_data(x_line, y_line)

        m_r  = round(float(m), 3)
        b_r  = round(float(b), 3)
        sign = "+" if b_r >= 0 else "-"
        title.set_text(f"Epoch {epoch} / {len(m_history)} | Loss: {round(loss, 4)}")
        eq_text.set_text(f"y = {m_r}*x {sign} {abs(b_r)}")
        
        # Flash a warning at the end if not converged
        if frame_idx == len(frame_indices) - 1 and not self._history.converged:
            ax.text(0.5, 0.5, "WARNING: Loss still decreasing!\nIncrease epochs or learning rate.", 
                    transform=ax.transAxes, ha="center", va="center", 
                    fontsize=12, color="red", alpha=0.8, weight="bold")
            
        return line, title, eq_text

    anim = animation.FuncAnimation(
        fig, _update,
        frames=len(frame_indices),
        interval=interval,
        blit=False,
    )

    plt.tight_layout()

    if display:
        return self.show(animation_obj=anim)
    return anim
