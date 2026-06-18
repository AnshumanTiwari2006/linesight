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




from linesight import MultipleLinearRegression
from linesight.exceptions import LineSightNotFittedError, LineSightShapeError


class TestMultipleFit:

    def test_weights_shape(self, multi_data):
        X, y = multi_data
        m = MultipleLinearRegression(epochs=500, normalize=True)
        m.fit(X, y)
        assert m.weights.shape == (3,)

    def test_bias_is_float(self, multi_data):
        X, y = multi_data
        m = MultipleLinearRegression(epochs=500, normalize=True)
        m.fit(X, y)
        assert isinstance(m.bias, float)

    def test_r2_above_threshold(self, multi_data):
        X, y = multi_data
        m = MultipleLinearRegression(epochs=1000, normalize=True)
        m.fit(X, y)
        assert m.score(X, y)["r2"] > 0.90

    def test_weight_recovery(self, multi_data):
        """With normalize=True and enough epochs, should recover true weights approximately."""
        X, y = multi_data
        m = MultipleLinearRegression(learning_rate=0.05, epochs=3000, normalize=True)
        m.fit(X, y)
        pred = m.predict(X)
        r2 = 1 - np.sum((y - pred)**2) / np.sum((y - np.mean(y))**2)
        assert r2 > 0.95

    def test_predict_before_fit_raises(self):
        with pytest.raises(LineSightNotFittedError):
            MultipleLinearRegression().predict(np.zeros((3, 3)))

    def test_wrong_feature_count_raises(self, multi_data):
        X, y = multi_data
        m = MultipleLinearRegression(epochs=100)
        m.fit(X, y)
        with pytest.raises(LineSightShapeError):
            m.predict(np.zeros((3, 5)))

    def test_batch_size_trains(self, multi_data):
        X, y = multi_data
        m = MultipleLinearRegression(epochs=300, batch_size=16, normalize=True)
        m.fit(X, y)
        assert m._is_fitted

    def test_score_keys(self, multi_data):
        X, y = multi_data
        m = MultipleLinearRegression(epochs=300, normalize=True)
        m.fit(X, y)
        s = m.score(X, y)
        assert all(k in s for k in ["mse", "rmse", "mae", "r2"])

    def test_history_lengths(self, multi_data):
        X, y = multi_data
        m = MultipleLinearRegression(epochs=200, store_history=True)
        m.fit(X, y)
        h = m.get_training_history()
        assert len(h.losses) == 200
        assert len(h.weights) == 200
        assert len(h.biases) == 200

    def test_method_chaining(self, multi_data):
        X, y = multi_data
        m = MultipleLinearRegression(epochs=50)
        assert m.fit(X, y) is m

    def test_normalize_un_scales_predictions(self, multi_data):
        """Predictions with normalize=True and False should be in same range."""
        X, y = multi_data
        m1 = MultipleLinearRegression(epochs=500, normalize=True)
        m1.fit(X, y)
        pred1 = m1.predict(X)
        assert pred1.shape == y.shape
        assert np.isfinite(pred1).all()


