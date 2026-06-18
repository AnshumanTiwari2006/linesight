from linesight.utils.validators import _validate_Xy
from linesight.metrics import accuracy

def score(self, X, y) -> dict:
    self._check_fitted("score")
    X, y = _validate_Xy(X, y)
    y_pred = self.predict(X)
    return {
        "accuracy": round(accuracy(y, y_pred), 4),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
    }
