import numpy as np
from linesight.utils.validators import _validate_X


def predict(self, X, threshold: float = 0.5) -> np.ndarray:
    """Predict class labels (0 or 1) by thresholding predict_proba()."""
    self._check_fitted("predict")
    proba = self.predict_proba(X)
    return (proba >= threshold).astype(int)
