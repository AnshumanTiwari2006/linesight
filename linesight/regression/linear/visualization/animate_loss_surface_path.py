import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import warnings
from linesight.utils.validators import _validate_Xy
from linesight.exceptions import LineSightShapeError, LineSightDataWarning
from linesight.utils.colors import GRADIENT_PATH, LOSS_FINAL_DOT


def animate_loss_surface_path(self, X, y, grid_points: int = 40,
                               skip_frames: int = 10,
                               interval: int = 60,
                               display: bool = True):
    """
    Animate the gradient descent optimization path on the 3D loss surface.

    The surface is computed once (expensive). Then each frame moves the
    red dot along the surface showing the (m, b) values at each epoch.

    Requires store_history=True and single-feature X.

    What is drawn
    -------------
    Frame 0: Loss surface rendered, red dot at starting position (m=0, b=0)
    Each frame: red dot moves to next (m, b) from training history
    Final frame: dot sits at the bottom of the bowl (the optimum)

    Title updates: "Epoch {n} — Loss: {loss}"

    Parameters
    ----------
    X : array-like, shape (n, 1)
    y : array-like, shape (n,)
    grid_points : int, default 40
        Surface resolution. Lower than plot_loss_surface default
        because the surface is rendered once per animation build (slow).
        Do not exceed 60 — the animation build will take >30 seconds.
    skip_frames : int, default 10
        Only animate every Nth epoch. 1000 epochs / 10 = 100 frames.
    interval : int, default 60
        Milliseconds between frames.
    display : bool, default True

    Raises
    ------
    LineSightShapeError
        If X has more than 1 feature:
        "animate_loss_surface_path() requires single-feature X.
        Your X has {n} features. The loss surface is 3D only for 1-feature models."

    LineSightDataWarning (warns, does not raise)
        If store_history=False:
        "No history available. Re-fit with store_history=True to animate the path."

    LineSightDataWarning (warns, does not raise)
        If grid_points > 60:
        "grid_points={n} will make the animation build very slow (>30s).
        Consider grid_points=40 (default)."
    """
    self._check_fitted("animate_loss_surface_path")

    if self._history.is_empty():
        warnings.warn(
            "No history available. Re-fit with store_history=True.",
            LineSightDataWarning, stacklevel=2
        )
        return None

    X, y = _validate_Xy(X, y)
    if X.shape[1] != 1:
        raise LineSightShapeError(
            f"animate_loss_surface_path() requires single-feature X.\n"
            f"Your X has {X.shape[1]} features.\n"
            f"The loss surface is 3D only for 1-feature models (slope + intercept)."
        )

    if grid_points > 60:
        warnings.warn(
            f"grid_points={grid_points} will make animation build very slow (>30s).\n"
            f"Consider grid_points=40.",
            LineSightDataWarning, stacklevel=2
        )

    x = X.ravel()
    weights = self._history.weights
    biases = self._history.biases
    losses = self._history.losses

    # Select frames
    frame_indices = list(range(0, len(weights), skip_frames))
    if frame_indices[-1] != len(weights) - 1:
        frame_indices.append(len(weights) - 1)

    # Build loss surface grid ONCE
    m_fit = float(np.ravel(self.coef_)[0])
    b_fit = float(self.intercept_)
    span_m = max(abs(m_fit) * 3, 1.0)
    span_b = max(abs(b_fit) * 3, 1.0)

    m_vals = np.linspace(m_fit - span_m, m_fit + span_m, grid_points)
    b_vals = np.linspace(b_fit - span_b, b_fit + span_b, grid_points)
    M, B = np.meshgrid(m_vals, b_vals)

    x_3d = x[np.newaxis, np.newaxis, :]
    M_3d = M[:, :, np.newaxis]
    B_3d = B[:, :, np.newaxis]
    y_3d = y[np.newaxis, np.newaxis, :]
    Z = np.mean((M_3d * x_3d + B_3d - y_3d) ** 2, axis=2)

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(M, B, Z, cmap='Blues', alpha=0.5, linewidth=0)
    ax.set_xlabel("Slope (m)")
    ax.set_ylabel("Intercept (b)")
    ax.set_zlabel("Loss")
    title = ax.set_title("")

    # Initial dot position
    m0 = weights[0]
    b0 = biases[0]
    z0 = float(np.mean((m0 * x + b0 - y) ** 2))
    dot, = ax.plot([m0], [b0], [z0], 'o', color=LOSS_FINAL_DOT,
                   markersize=8, zorder=5)

    # Path line (builds up over frames)
    path_ms = [m0]
    path_bs = [b0]
    path_zs = [z0]
    path_line, = ax.plot(path_ms, path_bs, path_zs,
                         color=GRADIENT_PATH, linewidth=1.5, alpha=0.7)

    def _update(frame_num):
        idx = frame_indices[frame_num]
        b_val = biases[idx]
        m_val = weights[idx]
        z_val = losses[idx]
        epoch = idx + 1

        path_ms.append(m_val)
        path_bs.append(b_val)
        path_zs.append(z_val)

        dot.set_data_3d([m_val], [b_val], [z_val])
        path_line.set_data_3d(path_ms, path_bs, path_zs)
        title.set_text(f"Epoch {epoch} — Loss: {round(z_val, 4)}")
        return dot, path_line, title

    anim = animation.FuncAnimation(
        fig, _update, frames=len(frame_indices),
        interval=interval, blit=False
    )
    plt.tight_layout()

    if display:
        return self.show(animation_obj=anim)
    return anim
