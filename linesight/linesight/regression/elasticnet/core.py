import numpy as np
from linesight.base import LineSightBase
from linesight.utils.history import TrainingHistory

from linesight.regression.elasticnet.engine.fit import _fit
from linesight.regression.elasticnet.engine.predict import predict
from linesight.regression.elasticnet.engine.score import score
from linesight.regression.elasticnet.engine.get_history import get_training_history
from linesight.regression.elasticnet.explain.explain_regularization import explain_regularization
from linesight.regression.elasticnet.explain.explain_coefficients import explain_coefficients
from linesight.regression.elasticnet.visualization.plot_fit import plot_fit
from linesight.regression.elasticnet.visualization.plot_loss_curve import plot_loss_curve
from linesight.regression.elasticnet.visualization.plot_coefficient_shrinkage import plot_coefficient_shrinkage
from linesight.regression.elasticnet.visualization.compare_regularization_methods import compare_regularization_methods


from linesight.regression.elasticnet.visualization.plot_actual_vs_predicted import plot_actual_vs_predicted
from linesight.regression.elasticnet.visualization.plot_learning_curve import plot_learning_curve
from linesight.regression.elasticnet.visualization.plot_l1_l2_balance import plot_l1_l2_balance

from linesight.regression.elasticnet.visualization.animate_l1_ratio_sweep import animate_l1_ratio_sweep
from linesight.regression.linear.visualization.plot_residuals import plot_residuals

class ElasticNetRegression(LineSightBase):
    """
    ElasticNet regression: L1 (sparsity) + L2 (shrinkage) combined.
    l1_ratio=1 -> pure Lasso.  l1_ratio=0 -> pure Ridge.

    Parameters
    ----------
    alpha         : float, default 1.0
    l1_ratio      : float, default 0.5  (must be in [0, 1])
    epochs        : int,   default 1000
    store_history : bool,  default False
    """

    def __init__(
        self,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        epochs: int = 1000,
        store_history: bool = False,
        normalize: bool = False,
    ):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.epochs = epochs
        self.store_history = store_history
        self.normalize = normalize
        self.store_history = store_history
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
    compare_regularization_methods = compare_regularization_methods

    plot_actual_vs_predicted = plot_actual_vs_predicted
    plot_learning_curve = plot_learning_curve
    plot_l1_l2_balance = plot_l1_l2_balance

    animate_l1_ratio_sweep = animate_l1_ratio_sweep
    plot_residuals = plot_residuals
    def summary(self) -> str:
        self._check_fitted("summary")
        converged_str = "Yes" if self._history.converged else "No"
        final_loss = round(self._history.losses[-1], 6) if not self._history.is_empty() else "N/A"
        
        lines = [
            "=" * 50,
            f"LineSight — ElasticNetRegression",
            "=" * 50,
            "Training config:",
            f"  Alpha:          {getattr(self, 'alpha', 'N/A')}",
            f"  L1 ratio:       {getattr(self, 'l1_ratio', 'N/A')}",
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
