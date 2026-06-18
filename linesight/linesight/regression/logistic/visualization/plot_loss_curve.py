import matplotlib.pyplot as plt
import warnings
from linesight.exceptions import LineSightDataWarning

def plot_loss_curve(self, display: bool = True):
    self._check_fitted("plot_loss_curve")
    if self._history.is_empty():
        warnings.warn("Re-fit with store_history=True.", LineSightDataWarning, stacklevel=2)
        return None

    losses = self._history.losses
    epochs = list(range(1, len(losses) + 1))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, losses, color="#1a6fcc", linewidth=1.5)
    ax.scatter([epochs[-1]], [losses[-1]], color="#e05252", s=40, zorder=3)
    ax.annotate(f"  Final BCE: {round(losses[-1], 4)}", xy=(epochs[-1], losses[-1]),
                fontsize=9, color="#e05252")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary Cross-Entropy Loss")
    ax.set_title("Training Loss — Logistic Regression")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if display:
        return self.show(fig=fig)
    return fig
