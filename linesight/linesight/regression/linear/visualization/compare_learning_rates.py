import numpy as np
import matplotlib.pyplot as plt
from linesight.utils.validators import _validate_Xy


def compare_learning_rates(self, X, y,
                            learning_rates=None,
                            epochs: int = 200,
                            display: bool = True):
    """
    Train the same model with multiple learning rates and compare loss curves.

    What is drawn
    -------------
    One subplot per learning rate showing the loss curve.
    Each subplot annotated with a diagnosis:
      - "Converged" — loss flattened out, model found a good minimum
      - "Diverged" — loss went to NaN/inf, learning rate too high
      - "Still decreasing" — loss still falling, needs more epochs or higher LR
      - "Oscillating" — loss bouncing up and down, LR slightly too high

    Parameters
    ----------
    X : array-like
    y : array-like
    learning_rates : list of float, optional
        Default: [0.0001, 0.001, 0.01, 0.1]
    epochs : int, default 200
        Epochs to train for each learning rate comparison.
    display : bool, default True

    Side effects
    ------------
    Does NOT modify self. Creates temporary model instances internally.
    """
    self._check_fitted("compare_learning_rates")

    if learning_rates is None:
        learning_rates = [0.0001, 0.001, 0.01, 0.1]

    X, y = _validate_Xy(X, y)

    # Import here to avoid circular at top
    from linesight.regression.linear.core import LinearRegression

    n = len(learning_rates)
    cols = 2
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(10, 3.5 * rows))
    axes = axes.ravel()

    for i, lr in enumerate(learning_rates):
        tmp = LinearRegression(learning_rate=lr, epochs=epochs, store_history=True)
        tmp.fit(X, y)
        losses = tmp._history.losses

        ax = axes[i]

        if any(np.isnan(l) or np.isinf(l) for l in losses):
            # Diverged — only plot up to divergence point
            valid = [l for l in losses if not (np.isnan(l) or np.isinf(l))]
            ax.plot(valid, color="#e05252", linewidth=1.5)
            diagnosis = "Diverged (LR too high)"
            color = "#e05252"
        else:
            ax.plot(losses, color="#1a6fcc", linewidth=1.5)
            # Diagnose based on loss behavior
            if len(losses) >= 20:
                last_change = losses[-1] - losses[-10]
                if abs(last_change) < 1e-5:
                    diagnosis = "Converged"
                    color = "#2a9d4e"
                elif last_change > 0:
                    diagnosis = "Oscillating (LR slightly too high)"
                    color = "#e08c2a"
                else:
                    diagnosis = "Still decreasing (try more epochs)"
                    color = "#7047c4"
            else:
                diagnosis = ""
                color = "#333333"

        ax.set_title(f"LR = {lr}  ->  {diagnosis}", fontsize=9, color=color)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel("Loss", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Learning rate comparison", fontsize=12, y=1.01)
    plt.tight_layout()

    if display:
        return self.show(fig=fig)
    return fig
