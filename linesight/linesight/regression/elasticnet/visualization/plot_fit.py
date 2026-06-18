import numpy as np
import matplotlib.pyplot as plt
from linesight.utils.validators import _validate_Xy

def plot_fit(self, X, y, display: bool = True):
    self._check_fitted("plot_fit")
    X, y = _validate_Xy(X, y)
    y_pred = self.predict(X)

    if X.shape[1] == 1:
        x = X.ravel()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(x, y, color="#888888", edgecolors="white", s=60, zorder=3, label="Data")
        x_line = np.linspace(x.min(), x.max(), 200)
        y_line = self.predict(x_line.reshape(-1, 1))
        ax.plot(x_line, y_line, color="#1a6fcc", linewidth=2, label="ElasticNet fit")
        ax.set_xlabel(self.feature_names_in_[0])
        ax.set_ylabel("y")
        ax.set_title(f"ElasticNet Fit (alpha={self.alpha}, l1_ratio={self.l1_ratio})")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
    else:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(y, y_pred, color="#888888", edgecolors="white", s=50, alpha=0.7)
        lims = [min(y.min(), y_pred.min()), max(y.max(), y_pred.max())]
        ax.plot(lims, lims, color="#1a6fcc", linestyle="--", linewidth=1.5, label="Perfect fit")
        ax.set_xlabel("Actual y")
        ax.set_ylabel("Predicted ŷ")
        ax.set_title(f"ElasticNet: Actual vs Predicted (alpha={self.alpha})")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if display:
        return self.show(fig=fig)
    return fig
