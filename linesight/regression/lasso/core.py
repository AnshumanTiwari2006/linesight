import numpy as np
from linesight.base import LineSightBase
from linesight.utils.history import TrainingHistory

from linesight.regression.lasso.engine.fit import _fit
from linesight.regression.lasso.engine.predict import predict
from linesight.regression.lasso.engine.score import score
from linesight.regression.lasso.engine.get_history import get_training_history
from linesight.regression.lasso.explain.explain_regularization import explain_regularization
from linesight.regression.lasso.explain.explain_sparsity import explain_sparsity
from linesight.regression.lasso.explain.explain_coefficients import explain_coefficients
from linesight.regression.lasso.visualization.plot_fit import plot_fit
from linesight.regression.lasso.visualization.plot_loss_curve import plot_loss_curve
from linesight.regression.lasso.visualization.plot_coefficient_shrinkage import plot_coefficient_shrinkage
from linesight.regression.lasso.visualization.plot_constraint_region import plot_constraint_region
from linesight.regression.lasso.visualization.plot_feature_elimination import plot_feature_elimination
from linesight.regression.lasso.visualization.compare_with_linear import compare_with_linear


from linesight.regression.lasso.visualization.plot_actual_vs_predicted import plot_actual_vs_predicted
from linesight.regression.lasso.visualization.plot_learning_curve import plot_learning_curve
from linesight.regression.lasso.visualization.animate_coordinate_descent import animate_coordinate_descent

from linesight.regression.lasso.visualization.plot_sparsity_path import plot_sparsity_path
from linesight.regression.linear.visualization.plot_residuals import plot_residuals

class LassoRegression(LineSightBase):
    """
    Lasso regression (L1 regularization) with coordinate descent.
    Uses soft-thresholding — creates exact zeros (feature selection).

    Parameters
    ----------
    alpha         : float, default 1.0
    epochs        : int,   default 1000
    store_history : bool,  default False
    """

    def __init__(
        self,
        alpha: float = 1.0,
        epochs: int = 1000,
        store_history: bool = False,
        normalize: bool = False,
    ):
        self.alpha = alpha
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
    explain_sparsity = explain_sparsity
    explain_coefficients = explain_coefficients
    plot_fit = plot_fit
    plot_loss_curve = plot_loss_curve
    plot_coefficient_shrinkage = plot_coefficient_shrinkage
    plot_constraint_region = plot_constraint_region
    plot_feature_elimination = plot_feature_elimination
    compare_with_linear = compare_with_linear

    plot_actual_vs_predicted = plot_actual_vs_predicted
    plot_learning_curve = plot_learning_curve
    animate_coordinate_descent = animate_coordinate_descent

    plot_sparsity_path = plot_sparsity_path
    plot_residuals = plot_residuals
    def summary(self) -> str:
        self._check_fitted("summary")
        converged_str = "Yes" if self._history.converged else "No"
        final_loss = round(self._history.losses[-1], 6) if not self._history.is_empty() else "N/A"
        
        lines = [
            "=" * 50,
            f"LineSight — LassoRegression",
            "=" * 50,
            "Training config:",
            f"  Alpha (L1):     {getattr(self, 'alpha', 'N/A')}",
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
