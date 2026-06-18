import pytest, numpy as np, warnings



@pytest.fixture(scope="module")
def multi_data():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 3))
    y = X @ np.array([2.0, -1.0, 0.5]) + 0.3 + rng.normal(0, 0.1, 80)
    return X, y

@pytest.fixture(scope="module")
def binary_data():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((100, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(float)
    return X, y




from linesight import LogisticRegression
from linesight.exceptions import LineSightShapeError, LineSightNotFittedError


class TestLogistic:

    def test_non_binary_y_raises(self, binary_data):
        X, _ = binary_data
        y_bad = np.array([0, 1, 2] * 34, dtype=float)[:100]
        with pytest.raises(LineSightShapeError, match="0s and 1s"):
            LogisticRegression().fit(X, y_bad)

    def test_weights_shape(self, binary_data):
        X, y = binary_data
        m = LogisticRegression(epochs=300, normalize=True)
        m.fit(X, y)
        assert m.weights.shape == (2,)

    def test_accuracy_above_threshold(self, binary_data):
        X, y = binary_data
        m = LogisticRegression(learning_rate=0.1, epochs=500, normalize=True)
        m.fit(X, y)
        acc = m.score(X, y)["accuracy"]
        assert acc > 0.80

    def test_predict_proba_range(self, binary_data):
        X, y = binary_data
        m = LogisticRegression(epochs=300, normalize=True)
        m.fit(X, y)
        p = m.predict_proba(X)
        assert np.all(p >= 0) and np.all(p <= 1)

    def test_predict_is_binary(self, binary_data):
        X, y = binary_data
        m = LogisticRegression(epochs=300, normalize=True)
        m.fit(X, y)
        preds = m.predict(X)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_batch_size(self, binary_data):
        X, y = binary_data
        m = LogisticRegression(epochs=200, batch_size=16, normalize=True)
        m.fit(X, y)
        assert m._is_fitted

    def test_score_has_accuracy_key(self, binary_data):
        X, y = binary_data
        m = LogisticRegression(epochs=200, normalize=True)
        m.fit(X, y)
        s = m.score(X, y)
        assert "accuracy" in s

    def test_normalize_inverse_applied_in_predict(self, binary_data):
        """Predictions with normalize=True should not crash and be valid."""
        X, y = binary_data
        m = LogisticRegression(epochs=200, normalize=True)
        m.fit(X, y)
        p = m.predict_proba(X)
        assert p.shape == (100,)
        assert np.isfinite(p).all()

    def test_explain_boundary_string(self, binary_data):
        X, y = binary_data
        m = LogisticRegression(epochs=200, normalize=True)
        m.fit(X, y)
        s = m.explain_boundary()
        assert isinstance(s, str)
        assert "0 =" in s

    def test_before_fit_raises(self):
        with pytest.raises(LineSightNotFittedError):
            LogisticRegression().predict(np.zeros((3, 2)))


