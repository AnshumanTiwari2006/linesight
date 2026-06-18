import numpy as np


def explain_sparsity(self) -> str:
    """
    Print which features were set to zero and which survived Lasso selection.
    """
    self._check_fitted("explain_sparsity")

    coefs = self.weights
    names = self.feature_names_in_

    zero_features = [n for n, c in zip(names, coefs) if c == 0]
    nonzero_features = [(n, round(float(c), 4)) for n, c in zip(names, coefs) if c != 0]

    lines = [
        f"Lasso Sparsity Report (alpha={self.alpha})",
        "-" * 30,
        "",
        f"Features KEPT ({len(nonzero_features)} / {len(names)}):",
    ]
    for name, val in nonzero_features:
        lines.append(f"  {name}: {val}")

    lines += ["", f"Features ELIMINATED ({len(zero_features)} / {len(names)}):"]
    for name in zero_features:
        lines.append(f"  {name}: 0 (set to zero by L1 penalty)")

    if not zero_features:
        lines.append("  None — increase alpha to induce more sparsity.")

    output = "\n".join(lines)
    from linesight.utils.environment import _detect_environment
    if _detect_environment() == 'script':
        print(output)
    return output