import numpy as np
from linesight.base import LineSightBase
from linesight.utils.history import TrainingHistory

from linesight.regression.polynomial.engine.fit import _fit
from linesight.regression.polynomial.engine.predict import predict
from linesight.regression.multiple.engine.score import score
from linesight.regression.polynomial.engine.expand_features import _expand_features
from linesight.regression.polynomial.explain.show_equation import show_equation
from linesight.regression.polynomial.visualization.plot_fit import plot_fit
from linesight.regression.polynomial.visualization.compare_degrees import compare_degrees
from linesight.regression.polynomial.visualization.animate_degree_increase import animate_degree_increase
# Also reuse from linear:
from linesight.regression.linear.visualization.plot_residuals import plot_residuals
from linesight.regression.linear.visualization.plot_loss_curve import plot_loss_curve
from linesight.regression.polynomial.visualization.plot_actual_vs_predicted import plot_actual_vs_predicted
from linesight.regression.polynomial.visualization.plot_learning_curve import plot_learning_curve


from linesight.regression.polynomial.visualization.plot_basis_functions import plot_basis_functions

class PolynomialRegression(LineSightBase):
    """
    Polynomial regression for a single input feature.

    Learns: ŷ = b + a₁x + a₂x² + ... + aᵈxᵈ

    For multi-feature polynomial, expand features manually
    and use MultipleLinearRegression.

    Parameters
    ----------
    degree : int, default 2
        Highest polynomial power. degree=1 is identical to LinearRegression.
    learning_rate : float, default 0.001
        Lower than LinearRegression default because polynomial features
        have larger magnitudes (x^9 can be huge), so the gradient is larger.
    epochs : int, default 5000
        More epochs than linear because higher-degree fits need more steps.
    store_history : bool, default False
    """

    def __init__(
        self,
        degree: int = 2,
        learning_rate: float = 0.001,
        epochs: int = 5000,
        normalize: bool = False,
        store_history: bool = False,
    ):
        self.degree         = degree
        self.learning_rate  = learning_rate
        self.epochs         = epochs
        self.normalize      = normalize
        self.store_history  = store_history
        self.theta_         = None
        self._n_features_in_ = 1
        self._is_fitted     = False
        self._history       = TrainingHistory(learning_rate=learning_rate)

    fit                    = _fit
    predict                = predict
    score                  = score
    show_equation          = show_equation
    plot_fit               = plot_fit
    plot_residuals         = plot_residuals   # reused from linear
    plot_loss_curve        = plot_loss_curve  # reused from linear
    compare_degrees        = compare_degrees
    animate_degree_increase = animate_degree_increase

    plot_actual_vs_predicted = plot_actual_vs_predicted
    plot_learning_curve = plot_learning_curve
    plot_basis_functions = plot_basis_functions
    def summary(self) -> str:
        self._check_fitted("summary")
        converged_str = "Yes" if self._history.converged else "No"
        final_loss = round(self._history.losses[-1], 6) if not self._history.is_empty() else "N/A"
        lines = [
            "=" * 50,
            "LineSight — PolynomialRegression",
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
