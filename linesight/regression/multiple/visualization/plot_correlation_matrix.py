import numpy as np
import matplotlib.pyplot as plt
from linesight.utils.validators import _validate_Xy


def plot_correlation_matrix(self, X, y, display: bool = True):
    """
    Heatmap of the correlation matrix for all features + target.
    """
    self._check_fitted("plot_correlation_matrix")
    X, y = _validate_Xy(X, y)

    data = np.column_stack([X, y])
    labels = self.feature_names_in_ + ["y"]
    corr = np.corrcoef(data.T)

    fig, ax = plt.subplots(figsize=(max(5, len(labels)), max(4, len(labels))))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha='center', va='center', fontsize=8)

    ax.set_title("Correlation matrix")
    plt.tight_layout()

    if display:
        return self.show(fig=fig)
    return fig
