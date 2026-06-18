import numpy as np
import matplotlib.pyplot as plt
from linesight.utils.validators import _validate_Xy


def plot_probability_surface(self, X, y, display: bool = True):
    """
    3D scatter + sigmoid probability surface for 2-feature logistic models.
    """
    self._check_fitted("plot_probability_surface")
    X, y = _validate_Xy(X, y)

    if X.shape[1] != 2:
        raise ValueError("plot_probability_surface requires exactly 2 features.")

    from mpl_toolkits.mplot3d import Axes3D

    x0 = np.linspace(X[:, 0].min(), X[:, 0].max(), 35)
    x1 = np.linspace(X[:, 1].min(), X[:, 1].max(), 35)
    X0, X1 = np.meshgrid(x0, x1)
    X_grid = np.c_[X0.ravel(), X1.ravel()]
    Z = self.predict_proba(X_grid).reshape(X0.shape)

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X0, X1, Z, alpha=0.55, cmap="coolwarm")
    ax.scatter(X[:, 0], X[:, 1], y, c=y, cmap="coolwarm",
               s=40, edgecolors="k", zorder=5, label="Data")
    ax.set_xlabel(self.feature_names_in_[0])
    ax.set_ylabel(self.feature_names_in_[1])
    ax.set_zlabel("P(y=1 | x)")
    ax.set_title("Probability Surface")
    plt.tight_layout()

    if display:
        return self.show(fig=fig)
    return fig
