import numpy as np
from linesight.base import LineSightBase
from linesight.utils.history import TrainingHistory

from linesight.regression.multiple.engine.fit import _fit
from linesight.regression.multiple.engine.predict import predict
from linesight.regression.multiple.engine.score import score
from linesight.regression.multiple.engine.get_history import get_training_history
from linesight.regression.multiple.explain.show_equation import show_equation
from linesight.regression.multiple.explain.explain_coefficients import explain_coefficients
from linesight.regression.multiple.visualization.plot_fit import plot_fit
from linesight.regression.multiple.visualization.plot_feature_importance import plot_feature_importance
from linesight.regression.multiple.visualization.plot_partial_regression import plot_partial_regression
from linesight.regression.multiple.visualization.plot_correlation_matrix import plot_correlation_matrix
from linesight.regression.multiple.visualization.plot_prediction_plane import plot_prediction_plane


from linesight.regression.multiple.visualization.plot_actual_vs_predicted import plot_actual_vs_predicted
from linesight.regression.multiple.visualization.plot_learning_curve import plot_learning_curve

from linesight.regression.multiple.visualization.plot_residual_heatmap import plot_residual_heatmap
from linesight.regression.multiple.visualization.plot_multicollinearity import plot_multicollinearity
from linesight.regression.multiple.visualization.plot_3d_loss_slice import plot_3d_loss_slice

class MultipleLinearRegression(LineSightBase):
    """
    Multiple linear regression with vectorized gradient descent.
    y_pred = dot(X, weights) + bias

    Parameters
    ----------
    learning_rate : float, default 0.01
    epochs        : int,   default 1000
    store_history : bool,  default False
    batch_size    : int | None, default None (full-batch)
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        epochs: int = 1000,
        store_history: bool = False,
        batch_size: int = None,
        normalize: bool = False,
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.store_history = store_history
        self.batch_size = batch_size
        self.normalize = normalize
        self.weights = np.zeros(1)
        self.bias = 0.0
        self._history = TrainingHistory()
        self._n_features_in_ = 1
        self._is_fitted = False

    @property
    def coef_(self):
        return self.weights if hasattr(self, "weights") else np.array([])

    @property
    def intercept_(self):
        return float(self.bias) if hasattr(self, "bias") else 0.0

    fit = _fit
    predict = predict
    score = score
    get_training_history = get_training_history
    show_equation = show_equation
    explain_coefficients = explain_coefficients
    plot_fit = plot_fit
    plot_feature_importance = plot_feature_importance
    plot_partial_regression = plot_partial_regression
    plot_correlation_matrix = plot_correlation_matrix
    plot_prediction_plane = plot_prediction_plane

    plot_actual_vs_predicted = plot_actual_vs_predicted
    plot_learning_curve = plot_learning_curve

    plot_residual_heatmap = plot_residual_heatmap
    plot_multicollinearity = plot_multicollinearity
    plot_3d_loss_slice = plot_3d_loss_slice
    def summary(self) -> str:
        self._check_fitted("summary")
        converged_str = "Yes" if self._history.converged else "No"
        final_loss = round(self._history.losses[-1], 6) if not self._history.is_empty() else "N/A"
        
        lines = [
            "=" * 50,
            f"LineSight — MultipleLinearRegression",
            "=" * 50,
            "Training config:",
            f"  Learning rate:  {getattr(self, 'learning_rate', 'N/A')}",
            f"  Epochs:         {getattr(self, 'epochs', 'N/A')}",
            f"  Converged:      {converged_str}",
            f"  Final loss:     {final_loss}",
            "=" * 50,
        ]
        output = "\n".join(lines)
        from linesight.utils.environment import _detect_environment
        if _detect_environment() == 'script':
            print(output)
        return output
