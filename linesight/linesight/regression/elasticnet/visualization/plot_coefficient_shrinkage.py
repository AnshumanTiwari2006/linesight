import numpy as np
import matplotlib.pyplot as plt
import warnings
from linesight.utils.validators import _validate_Xy

def plot_coefficient_shrinkage(self, X, y, alphas=None, display: bool = True):
    """ElasticNet regularization path."""
    self._check_fitted("plot_coefficient_shrinkage")
    X, y = _validate_Xy(X, y)

    if alphas is None:
        alphas = np.logspace(-3, 1, 50)

    from linesight.regression.elasticnet.core import ElasticNetRegression
    n_features = X.shape[1]
    coef_paths = np.zeros((len(alphas), n_features))

    for i, a in enumerate(alphas):
        tmp = ElasticNetRegression(alpha=a, l1_ratio=self.l1_ratio, epochs=self.epochs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tmp.fit(X, y)
        coef_paths[i] = tmp.weights

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, n_features))

    for j, (name, color) in enumerate(zip(self.feature_names_in_, colors)):
        ax.plot(alphas, coef_paths[:, j], color=color, linewidth=1.5, label=name)

    ax.axvline(self.alpha, color="#e05252", linestyle="--", linewidth=1,
               label=f"Current alpha={self.alpha}")
    ax.set_xscale("log")
    ax.set_xlabel("alpha (log scale)")
    ax.set_ylabel("Coefficient value")
    ax.set_title(f"ElasticNet coefficient path (l1_ratio={self.l1_ratio})")
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if display:
        return self.show(fig=fig)
    return fig
