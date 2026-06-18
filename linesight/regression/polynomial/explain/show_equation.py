def show_equation(self) -> str:
    """
    Display the learned polynomial equation.

    Example output for degree=3:
        ŷ = 1.2000 + 3.4200x¹ - 0.8100x² + 0.0340x³

    Superscript formatting uses unicode characters for readability.
    """
    self._check_fitted("show_equation")

    superscripts = {
        1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵",
        6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹"
    }

    terms = [f"{round(float(self.theta_[0]), 4)}"]
    for p in range(1, self.degree + 1):
        coef = round(float(self.theta_[p]), 4)
        sup = superscripts.get(p, f"^{p}")
        sign = "+" if coef >= 0 else "-"
        terms.append(f"{sign} {abs(coef)}x{sup}")

    eq = "ŷ = " + " ".join(terms)

    from linesight.utils.environment import _detect_environment
    if _detect_environment() == 'script':
        print(eq)
    return eq
