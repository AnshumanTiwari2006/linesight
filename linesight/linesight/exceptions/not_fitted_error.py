class LineSightNotFittedError(RuntimeError):
    """
    Raised when any method that requires a trained model is called before fit().

    Always constructed with the method name that was called.

    Example
    -------
    raise LineSightNotFittedError(
        "Call fit(X, y) before calling predict()."
    )
    """
    pass
