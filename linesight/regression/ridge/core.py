import numpy as np
from linesight.base import LineSightBase
from linesight.utils.history import TrainingHistory

from linesight.regression.ridge.engine.fit import _fit
from linesight.regression.ridge.engine.predict import predict
from linesight.regression.ridge.engine.score import score
from linesight.regression.ridge.engine.get_history import get_training_history
from linesight.regression.ridge.explain.explain_regularization import explain_regularization
from linesight.regression.ridge.explain.explain_coefficients import explain_coefficients
from linesight.regression.ridge.visualization.plot_fit import plot_fit
from linesight.regression.ridge.visualization.plot_loss_curve import plot_loss_curve
from linesight.regression.ridge.visualization.plot_coefficient_shrinkage import plot_coefficient_shrinkage
from linesight.regression.ridge.visualization.plot_constraint_region import plot_constraint_region
from linesight.regression.ridge.visualization.compare_with_linear import compare_with_linear


from linesight.regression.ridge.visualization.plot_actual_vs_predicted import plot_actual_vs_predicted
from linesight.regression.ridge.visualization.plot_learning_curve import plot_learning_curve
from linesight.regression.ridge.visualization.plot_bias_variance_tradeoff import plot_bias_variance_tradeoff
from linesight.regression.ridge.visualization.animate_regularization import animate_regularization

from linesight.regression.ridge.visualization.plot_effective_degrees_of_freedom import plot_effective_degrees_of_freedom
from linesight.regression.linear.visualization.plot_residuals import plot_residuals

class RidgeRegression(LineSightBase):
    """
    Ridge regression (L2 regularization) with gradient descent.
    Loss = MSE + alpha * sum(weights^2)
    The bias is NEVER penalized.

    Parameters
    ----------
    alpha         : float, default 1.0
    learning_rate : float, default 0.01
    epochs        : int,   default 1000
    store_history : bool,  default False
    batch_size    : int | None, default None
    """

    def __init__(
        self,
        alpha: float = 1.0,
        learning_rate: float = 0.01,
        epochs: int = 1000,
        store_history: bool = False,
        batch_size: int = None,
        normalize: bool = False,
    ):
        self.alpha = alpha
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
        return self.weights

    @property
    def intercept_(self):
        return float(self.bias)

    fit = _fit
    predict = predict
    score = score
    get_training_history = get_training_history
    explain_regularization = explain_regularization
    explain_coefficients = explain_coefficients
    plot_fit = plot_fit
    plot_loss_curve = plot_loss_curve
    plot_coefficient_shrinkage = plot_coefficient_shrinkage
    plot_constraint_region = plot_constraint_region
    compare_with_linear = compare_with_linear

    plot_actual_vs_predicted = plot_actual_vs_predicted
    plot_learning_curve = plot_learning_curve
    plot_bias_variance_tradeoff = plot_bias_variance_tradeoff
    animate_regularization = animate_regularization

    plot_effective_degrees_of_freedom = plot_effective_degrees_of_freedom
    plot_residuals = plot_residuals
    def summary(self) -> str:
        self._check_fitted("summary")
        converged_str = "Yes" if self._history.converged else "No"
        final_loss = round(self._history.losses[-1], 6) if not self._history.is_empty() else "N/A"
        
        lines = [
            "=" * 50,
            f"LineSight — RidgeRegression",
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
