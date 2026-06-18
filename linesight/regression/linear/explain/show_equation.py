def show_equation(self) -> str:
    """
    Print and return the learned regression equation as a human-readable string.

    Output format:  y = 3.4200x + 1.2000
    Negative intercept:  y = 3.4200x - 0.8000

    Returns
    -------
    str — the equation string (also printed to stdout)
    """
    self._check_fitted("show_equation")

    m = round(float(self.m), 4)
    b = round(float(self.b), 4)

    if b >= 0:
        eq = f"y = {m}x + {b}"
    else:
        eq = f"y = {m}x - {abs(b)}"

    from linesight.utils.environment import _detect_environment
    if _detect_environment() == 'script':
        print(eq)
    return eq