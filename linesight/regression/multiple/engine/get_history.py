import warnings
from linesight.exceptions import LineSightDataWarning


def get_training_history(self):
    """Return the TrainingHistory from the last fit() call."""
    self._check_fitted("get_training_history")
    if self._history.is_empty():
        warnings.warn(
            "Training history is empty. Re-fit with store_history=True.",
            LineSightDataWarning, stacklevel=2
        )
    return self._history
