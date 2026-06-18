class LineSightShapeError(ValueError):
    """
    Raised when array shapes are incompatible.

    Always constructed with a message that includes:
    - What was expected
    - What was actually received
    - A suggested fix

    Example
    -------
    raise LineSightShapeError(
        "X and y must have the same number of samples.\n"
        f"X has {X.shape[0]} samples, y has {y.shape[0]} samples.\n"
        "Did you forget to align your datasets?"
    )
    """
    pass
