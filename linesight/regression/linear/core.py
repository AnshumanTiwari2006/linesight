import numpy as np
from linesight.base import LineSightBase
from linesight.utils.history import TrainingHistory

from linesight.regression.linear.engine.fit import _fit
from linesight.regression.linear.engine.predict import predict
from linesight.regression.linear.engine.score import score
from linesight.regression.linear.engine.get_history import get_training_history
from linesight.regression.linear.explain.show_equation import show_equation
from linesight.regression.linear.explain.explain_coefficients import explain_coefficients
from linesight.regression.linear.explain.explain_fit import explain_fit
from linesight.regression.linear.visualization.plot_fit import plot_fit
from linesight.regression.linear.visualization.plot_residuals import plot_residuals
from linesight.regression.linear.visualization.plot_loss_curve import plot_loss_curve
from linesight.regression.linear.visualization.animate_training import animate_training
from linesight.regression.linear.visualization.plot_loss_surface import plot_loss_surface
from linesight.regression.linear.visualization.compare_learning_rates import compare_learning_rates


from linesight.regression.linear.visualization.plot_actual_vs_predicted import plot_actual_vs_predicted
from linesight.regression.linear.visualization.plot_learning_curve import plot_learning_curve

from linesight.regression.linear.visualization.plot_gradient_vectors import plot_gradient_vectors
from linesight.regression.linear.visualization.plot_prediction_intervals import plot_prediction_intervals
from linesight.regression.linear.visualization.animate_loss_surface_path import animate_loss_surface_path
from linesight.regression.linear.visualization.plot_sensitivity_analysis import plot_sensitivity_analysis

class LinearRegression(LineSightBase):
    """
    Single-feature linear regression with gradient descent.

    Internal representation : m (slope) and b (intercept).
    Unified interface       : .weights  ->  np.array([m])
                               .bias     ->  b

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
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.store_history = store_history
        self.batch_size = batch_size

        # Tutor-style internal names
        self.m = 0.0
        self.b = 0.0

        self._n_features_in_ = 1
        self.feature_names_in_ = ["x0"]
        self._is_fitted = False
        self._history = TrainingHistory(learning_rate=learning_rate, epochs_run=0)

    # ── Unified interface (same API as all other models) ──────────────────────
    @property
    def weights(self) -> np.ndarray:
        """Alias so cross-model code calling .weights always works."""
        return np.array([self.m])

    @weights.setter
    def weights(self, value):
        self.m = float(np.asarray(value).ravel()[0])

    @property
    def bias(self) -> float:
        """Alias so cross-model code calling .bias always works."""
        return self.b

    @bias.setter
    def bias(self, value):
        self.b = float(value)

    # ── sklearn-style aliases ─────────────────────────────────────────────────
    @property
    def coef_(self):
        return np.array([self.m])

    @property
    def intercept_(self):
        return self.b

    fit = _fit
    predict = predict
    score = score
    get_training_history = get_training_history
    show_equation = show_equation
    explain_coefficients = explain_coefficients
    explain_fit = explain_fit
    plot_fit = plot_fit
    plot_residuals = plot_residuals
    plot_loss_curve = plot_loss_curve
    animate_training = animate_training
    plot_loss_surface = plot_loss_surface
    compare_learning_rates = compare_learning_rates

    plot_actual_vs_predicted = plot_actual_vs_predicted
    plot_learning_curve = plot_learning_curve

    plot_gradient_vectors = plot_gradient_vectors
    plot_prediction_intervals = plot_prediction_intervals
    animate_loss_surface_path = animate_loss_surface_path
    plot_sensitivity_analysis = plot_sensitivity_analysis
    def summary(self) -> str:
        self._check_fitted("summary")
        converged_str = "Yes" if self._history.converged else "No"
        final_loss = round(self._history.losses[-1], 6) if not self._history.is_empty() else "N/A"
        
        lines = [
            "=" * 50,
            f"LineSight — LinearRegression",
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
