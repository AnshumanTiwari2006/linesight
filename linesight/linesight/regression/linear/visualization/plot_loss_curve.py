import matplotlib.pyplot as plt
import warnings
from linesight.exceptions import LineSightDataWarning
from linesight.utils.viz_context import print_viz_context


def plot_loss_curve(self, display: bool = True):
    """
    Plot the training loss over epochs.

    Requires store_history=True during fit().

    What is drawn
    -------------
    - X-axis: epoch number
    - Y-axis: MSE loss
    - Final loss value annotated on the last point
    - The curve shape teaches: fast initial drop, then slow convergence

    Parameters
    ----------
    display : bool, default True

    Warns
    -----
    LineSightDataWarning if store_history=False
    """
    self._check_fitted("plot_loss_curve")

    if self._history.is_empty():
        warnings.warn(
            "No history to plot. Re-fit with store_history=True.",
            LineSightDataWarning, stacklevel=2
        )
        return None

    losses = self._history.losses
    epochs = list(range(1, len(losses) + 1))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, losses, color="#1a6fcc", linewidth=1.5)
    ax.scatter([epochs[-1]], [losses[-1]], color="#e05252", s=40, zorder=3)
    ax.annotate(f"  Final: {round(losses[-1], 4)}",
                xy=(epochs[-1], losses[-1]),
                fontsize=9, color="#e05252")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("Training loss over epochs")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    print_viz_context(
        diagram="Training Loss Curve",
        solves="Shows whether gradient descent is converging and how fast.",
        theory=(
            "At each epoch, gradient descent updates weights using dJ/dw."
            " The MSE loss should strictly decrease every epoch when the"
            " learning rate is appropriate. If it flattens early, more epochs"
            " or a larger learning rate is needed. If it explodes, the learning"
            " rate is too high."
        ),
        formula="J = (1/n) * sum((y_i - y_hat_i)^2)   |   dJ/dm = (2/n)*sum((y_hat-y)*x)   dJ/db = (2/n)*sum(y_hat-y)",
        constraints=(
            "- Requires store_history=True during .fit()\n"
            "- One loss point recorded per epoch\n"
            "- Loss should always be >= 0 (MSE cannot go negative)"
        ),
        reading=(
            "- Steep initial drop = fast early learning (large gradient).\n"
            "- Gradual flattening = approaching the minimum (converging).\n"
            "- Completely flat early = learning rate too small or data is trivial.\n"
            "- Spike or divergence upward = learning rate too high, reduce it."
        ),
    )
    if display:
        return self.show(fig=fig)
    return fig
