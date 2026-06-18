import numpy as np
import matplotlib.pyplot as plt
from linesight.utils.validators import _validate_Xy

def plot_sigmoid(self, X, y, display: bool = True):
    self._check_fitted("plot_sigmoid")
    X, y = _validate_Xy(X, y)
    z = np.dot(X, self.weights) + self.bias
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.scatter(z, y, color="#888888", edgecolors="white", s=60, zorder=3, alpha=0.6, label="Actual Data")
    
    z_line = np.linspace(z.min() - 1, z.max() + 1, 200)
    
    # manually apply sigmoid
    p_line = np.where(z_line >= 0, 1.0 / (1.0 + np.exp(-z_line)), np.exp(z_line) / (1.0 + np.exp(z_line)))
    
    ax.plot(z_line, p_line, color="#1a6fcc", linewidth=2, label="Sigmoid curve")
    ax.axhline(0.5, color="#e05252", linestyle="--", alpha=0.7, label="Decision Boundary (p=0.5)")
    ax.axvline(0, color="#e05252", linestyle="--", alpha=0.7)
    
    ax.set_xlabel("Linear Score (z = X@theta)")
    ax.set_ylabel("Probability p(y=1|x)")
    ax.set_title("Logistic Regression Fit")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if display:
        return self.show(fig=fig)
    return fig
