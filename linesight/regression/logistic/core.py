import numpy as np
from linesight.base import LineSightBase
from linesight.utils.history import TrainingHistory

from linesight.regression.logistic.engine.fit import _fit
from linesight.regression.logistic.engine.predict import predict
from linesight.regression.logistic.engine.predict_proba import predict_proba
from linesight.regression.logistic.engine.score import score
from linesight.regression.logistic.engine.get_history import get_training_history
from linesight.regression.logistic.explain.explain_boundary import explain_boundary
from linesight.regression.logistic.explain.explain_coefficients import explain_coefficients
from linesight.regression.logistic.explain.explain_sigmoid import explain_sigmoid
from linesight.regression.logistic.visualization.plot_decision_boundary import plot_decision_boundary
from linesight.regression.logistic.visualization.plot_loss_curve import plot_loss_curve
from linesight.regression.logistic.visualization.plot_probability_surface import plot_probability_surface
from linesight.regression.logistic.visualization.plot_sigmoid import plot_sigmoid
from linesight.regression.logistic.visualization.animate_boundary import animate_boundary


from linesight.regression.logistic.visualization.plot_actual_vs_predicted import plot_actual_vs_predicted
from linesight.regression.logistic.visualization.plot_learning_curve import plot_learning_curve
from linesight.regression.logistic.visualization.plot_confusion_matrix import plot_confusion_matrix
from linesight.regression.logistic.visualization.plot_roc_curve import plot_roc_curve
from linesight.regression.logistic.visualization.plot_threshold_sensitivity import plot_threshold_sensitivity

from linesight.regression.logistic.visualization.plot_log_odds import plot_log_odds
from linesight.regression.logistic.visualization.plot_calibration_curve import plot_calibration_curve
from linesight.regression.logistic.visualization.plot_residuals import plot_residuals

class LogisticRegression(LineSightBase):
    """
    Binary logistic regression with gradient descent.
    Uses sigmoid -> binary cross-entropy loss.
    y must contain only 0s and 1s.

    Parameters
    ----------
    learning_rate : float, default 0.01
    epochs        : int,   default 1000
    store_history : bool,  default False
    batch_size    : int | None, default None
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
        return self.weights

    @property
    def intercept_(self):
        return float(self.bias)

    fit = _fit
    predict = predict
    predict_proba = predict_proba
    score = score
    get_training_history = get_training_history
    explain_boundary = explain_boundary
    explain_coefficients = explain_coefficients
    explain_sigmoid = explain_sigmoid
    plot_decision_boundary = plot_decision_boundary
    plot_loss_curve = plot_loss_curve
    plot_probability_surface = plot_probability_surface
    plot_sigmoid = plot_sigmoid
    animate_boundary = animate_boundary

    plot_actual_vs_predicted = plot_actual_vs_predicted
    plot_learning_curve = plot_learning_curve
    plot_confusion_matrix = plot_confusion_matrix
    plot_roc_curve = plot_roc_curve
    plot_threshold_sensitivity = plot_threshold_sensitivity

    plot_log_odds = plot_log_odds
    plot_calibration_curve = plot_calibration_curve
    plot_residuals = plot_residuals
    def summary(self) -> str:
        self._check_fitted("summary")
        converged_str = "Yes" if self._history.converged else "No"
        final_loss = round(self._history.losses[-1], 6) if not self._history.is_empty() else "N/A"
        
        lines = [
            "=" * 50,
            f"LineSight — LogisticRegression",
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
