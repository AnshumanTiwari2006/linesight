import numpy as np
import matplotlib.pyplot as plt
from linesight.utils.validators import _validate_Xy
from linesight.utils.colors import DATA_POINTS


def compare_degrees(self, X, y, degrees=None, display: bool = True):
    """
    THIS IS THE MOST EDUCATIONAL VISUALIZATION IN THE ENTIRE LIBRARY.

    Fit the same data with multiple polynomial degrees and show all fits
    side by side. Makes overfitting and underfitting immediately visible.

    What is drawn
    -------------
    One subplot per degree. Each shows:
    - Scatter of data points
    - The polynomial curve for that degree
    - R² score in the title
    - A qualitative label: "Underfit" / "Good fit" / "Overfit"

    The labels are determined by:
    - R² < 0.6  → "Underfit"
    - R² >= 0.6 and R² < 0.98 → "Good fit"
    - R² >= 0.98 → "Overfit (probably)" — because very high R² on training
      data with high degree usually means the model memorized the noise

    This distinction is heuristic and documented as such. The student
    should verify on validation data. The label plants the concept.

    Parameters
    ----------
    X : array-like
    y : array-like
    degrees : list of int, default [1, 2, 3, 5, 9]
    display : bool, default True

    Side effects
    ------------
    Creates temporary PolynomialRegression instances. Does not modify self.
    """
    if degrees is None:
        degrees = [1, 2, 3, 5, 9]

    X, y = _validate_Xy(X, y)
    x = X.ravel()
    x_curve = np.linspace(x.min(), x.max(), 300)

    from linesight.regression.polynomial.core import PolynomialRegression
    from linesight.metrics import r2

    n = len(degrees)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.ravel()

    degree_colors = [
        "#1a6fcc", "#2a9d4e", "#e05252", "#7047c4", "#e08c2a",
        "#d9534a", "#5280e0", "#888888", "#333333"
    ]

    for i, deg in enumerate(degrees):
        tmp = PolynomialRegression(
            degree=deg,
            learning_rate=self.learning_rate,
            epochs=self.epochs,
            normalize=True,
        )
        tmp.fit(X, y)

        y_pred = tmp.predict(X)
        r2_val = round(r2(y, y_pred), 3)
        y_curve = tmp.predict(x_curve.reshape(-1, 1))

        if r2_val < 0.6:
            quality = "Underfit"
            title_color = "#e08c2a"
        elif r2_val >= 0.98:
            quality = "Overfit (probably)"
            title_color = "#e05252"
        else:
            quality = "Good fit"
            title_color = "#2a9d4e"

        ax = axes[i]
        color = degree_colors[i % len(degree_colors)]

        ax.scatter(x, y, color=DATA_POINTS, edgecolors="white",
                   linewidths=0.6, s=40, zorder=3)
        ax.plot(x_curve, y_curve, color=color, linewidth=2)
        ax.set_title(f"Degree {deg}  —  R²={r2_val}\n{quality}",
                     fontsize=10, color=title_color)
        ax.set_xlabel("X")
        ax.set_ylabel("ŷ")
        ax.spines[["top", "right"]].set_visible(False)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Polynomial degree comparison — underfitting vs overfitting",
                 fontsize=12, y=1.01)
    plt.tight_layout()

    if display:
        return self.show(fig=fig)
    return fig
