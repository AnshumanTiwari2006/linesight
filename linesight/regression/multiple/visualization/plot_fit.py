import numpy as np
import matplotlib.pyplot as plt
from linesight.utils.validators import _validate_Xy


def plot_fit(self, X, y, display: bool = True):
    """
    For 1-feature: scatter + line. For 2-feature: 3D surface + scatter.
    For >2 features: actual vs predicted scatter with diagonal identity line.
    """
    self._check_fitted("plot_fit")
    X, y = _validate_Xy(X, y)
    y_pred = self.predict(X)
    n_features = X.shape[1]

    if n_features == 1:
        x = X.ravel()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(x, y, color="#888888", edgecolors="white", s=60, zorder=3, label="Data")
        x_line = np.linspace(x.min(), x.max(), 200)
        X_line = x_line.reshape(-1, 1)
        y_line = self.predict(X_line)
        ax.plot(x_line, y_line, color="#1a6fcc", linewidth=2, label="Fit")
        ax.set_xlabel(self.feature_names_in_[0])
        ax.set_ylabel("y")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)

    elif n_features == 2:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], y, color="#888888", s=40, label="Data")

        x0 = np.linspace(X[:, 0].min(), X[:, 0].max(), 30)
        x1 = np.linspace(X[:, 1].min(), X[:, 1].max(), 30)
        X0, X1 = np.meshgrid(x0, x1)
        X_grid = np.c_[X0.ravel(), X1.ravel()]
        Z = self.predict(X_grid).reshape(X0.shape)
        ax.plot_surface(X0, X1, Z, alpha=0.4, cmap='Blues')
        ax.set_xlabel(self.feature_names_in_[0])
        ax.set_ylabel(self.feature_names_in_[1])
        ax.set_zlabel("y")

    else:
        # Actual vs predicted
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(y, y_pred, color="#888888", edgecolors="white", s=50, alpha=0.7)
        lims = [min(y.min(), y_pred.min()), max(y.max(), y_pred.max())]
        ax.plot(lims, lims, color="#1a6fcc", linewidth=1.5, linestyle="--", label="Perfect fit")
        ax.set_xlabel("Actual y")
        ax.set_ylabel("Predicted ŷ")
        ax.set_title(f"Actual vs Predicted ({n_features} features)")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)

    ax.set_title(ax.get_title() or "Multiple Linear Regression Fit")
    plt.tight_layout()

    if display:
        return self.show(fig=fig)
    return fig
