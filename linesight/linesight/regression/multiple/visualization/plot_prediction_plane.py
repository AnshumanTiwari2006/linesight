import numpy as np
import matplotlib.pyplot as plt
from linesight.utils.validators import _validate_Xy
from linesight.exceptions import LineSightShapeError


def plot_prediction_plane(self, X, y, display: bool = True):
    """
    3D scatter + prediction plane. Only for exactly 2-feature models.
    """
    self._check_fitted("plot_prediction_plane")
    X, y = _validate_Xy(X, y)

    if X.shape[1] != 2:
        raise LineSightShapeError(
            f"plot_prediction_plane() requires exactly 2 features. "
            f"Your X has {X.shape[1]} features."
        )

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X[:, 0], X[:, 1], y, color="#888888", s=40, alpha=0.8, label="Data")

    x0 = np.linspace(X[:, 0].min(), X[:, 0].max(), 30)
    x1 = np.linspace(X[:, 1].min(), X[:, 1].max(), 30)
    X0, X1 = np.meshgrid(x0, x1)
    X_grid = np.c_[X0.ravel(), X1.ravel()]
    Z = self.predict(X_grid).reshape(X0.shape)

    ax.plot_surface(X0, X1, Z, alpha=0.4, cmap='Blues')
    ax.set_xlabel(self.feature_names_in_[0])
    ax.set_ylabel(self.feature_names_in_[1])
    ax.set_zlabel("y")
    ax.set_title("Prediction plane (2 features)")
    ax.legend()
    plt.tight_layout()

    if display:
        return self.show(fig=fig)
    return fig
