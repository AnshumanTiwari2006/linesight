import numpy as np
import matplotlib.pyplot as plt
from linesight.utils.validators import _validate_Xy
from linesight.exceptions import LineSightShapeError
from linesight.utils.viz_context import print_viz_context


def plot_decision_boundary(self, X, y, resolution: int = 200, display: bool = True):
    """
    Visualize the decision boundary for a 2-feature logistic regression model.

    What is drawn
    -------------
    1. Background color field: blue region = model predicts class 0,
       red region = model predicts class 1.
       Intensity of color = confidence (probability distance from 0.5).
       This is the "Desmos-like" visualization.
    2. The decision boundary: solid black line where P(y=1) = 0.5 exactly.
    3. Actual data points overlaid, colored by TRUE class (blue=0, red=1).
       Misclassified points get an X marker instead of a circle.

    Parameters
    ----------
    X : array-like, shape (n, 2) — MUST be exactly 2 features
    y : array-like, shape (n,) — binary labels 0 or 1
    resolution : int, default 200
        Grid points per axis. Higher = sharper boundary but slower.
        Use resolution=100 for animations.
    display : bool, default True

    Raises
    ------
    LineSightShapeError if X does not have exactly 2 features.
    """
    self._check_fitted("plot_decision_boundary")
    X, y = _validate_Xy(X, y)

    if X.shape[1] != 2:
        raise LineSightShapeError(
            f"plot_decision_boundary() requires exactly 2 features.\n"
            f"Your X has {X.shape[1]} features.\n"
            f"This is a fundamental visualization limit (we cannot plot in {X.shape[1]}D)."
        )

    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx1, xx2 = np.meshgrid(
        np.linspace(x1_min, x1_max, resolution),
        np.linspace(x2_min, x2_max, resolution)
    )
    grid = np.c_[xx1.ravel(), xx2.ravel()]
    proba = self.predict_proba(grid).reshape(xx1.shape)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Background probability field
    ax.contourf(xx1, xx2, proba, levels=50, cmap='RdBu_r', alpha=0.4,
                vmin=0, vmax=1)

    # Decision boundary line at p=0.5
    ax.contour(xx1, xx2, proba, levels=[0.5], colors='black', linewidths=1.5)

    # Data points
    for cls, color, label in [(0, '#4a90d9', 'Class 0'), (1, '#d9534a', 'Class 1')]:
        mask = y == cls
        ax.scatter(X[mask, 0], X[mask, 1], c=color, edgecolors='white',
                   linewidths=0.8, s=60, zorder=3, label=label)

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title("Decision boundary (line = P(class 1) = 0.50)")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    print_viz_context(
        diagram="Logistic Regression Decision Boundary",
        solves="Shows WHERE in 2D feature space the model splits classes, and how confidently.",
        theory=(
            "Logistic regression computes a linear score z = w1*x1 + w2*x2 + b,"
            " then maps it through the sigmoid: P(y=1) = 1 / (1 + exp(-z))."
            " The DECISION BOUNDARY is the line where P = 0.5, which means z = 0."
            " Solving for x2: x2 = -(w1*x1 + b) / w2  gives the exact boundary line."
            " Logistic regression ALWAYS creates a LINEAR boundary in the original feature space."
        ),
        formula=(
            "z = w1*x1 + w2*x2 + b   (log-odds)\n"
            "P(y=1) = sigmoid(z) = 1 / (1 + exp(-z))\n"
            "Boundary: z = 0  =>  x2 = -(w1*x1 + b) / w2"
        ),
        constraints=(
            "- Requires exactly 2 features (2D plot)\n"
            "- y must be binary (0 or 1 only)\n"
            "- The boundary is ALWAYS linear; use kernels for non-linear separation"
        ),
        reading=(
            "- Blue region = model predicts Class 0. Red region = Class 1.\n"
            "- Color intensity = confidence (darker = further from 50% boundary).\n"
            "- Black line = decision boundary (P = 0.5 exactly).\n"
            "- Points on the wrong side of the boundary = misclassified."
        ),
    )
    if display:
        return self.show(fig=fig)
    return fig
