import warnings
from linesight.exceptions import LineSightDataWarning


def get_training_history(self):
    """
    Return the TrainingHistory object from the last fit() call.

    Returns
    -------
    TrainingHistory dataclass with losses, coefficients, gradients lists.

    Raises
    ------
    LineSightNotFittedError if fit() has not been called.

    Warns
    -----
    LineSightDataWarning if store_history=False (history lists will be empty).
    """
    self._check_fitted("get_training_history")

    if self._history.is_empty():
        warnings.warn(
            "Training history is empty because store_history=False. "
            "Re-fit with store_history=True to access per-epoch data.",
            LineSightDataWarning,
            stacklevel=2
        )

    return self._history
