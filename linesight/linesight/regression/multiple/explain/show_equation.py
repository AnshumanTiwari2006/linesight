def show_equation(self) -> str:
    """Print the learned multivariate equation."""
    self._check_fitted("show_equation")

    intercept = round(float(self.bias), 4)
    coefs = [round(float(c), 4) for c in self.weights]
    names = self.feature_names_in_

    terms = " + ".join(f"{c}*{n}" for c, n in zip(coefs, names))
    eq = f"y = {intercept} + {terms}"
    from linesight.utils.environment import _detect_environment
    if _detect_environment() == 'script':
        print(eq)
    return eq