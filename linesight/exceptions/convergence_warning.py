import warnings

class LineSightConvergenceWarning(UserWarning):
    """
    Issued (not raised) when the model may not have converged.
    Use warnings.warn(..., LineSightConvergenceWarning) to issue it.
    Does not stop execution.
    """
    pass
