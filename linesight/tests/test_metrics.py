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




from linesight.metrics import mse, rmse, mae, r2, accuracy


class TestMetrics:

    def test_mse_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mse(y, y) == pytest.approx(0.0)

    def test_mse_known(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 2.0, 2.0])
        # errors: 1, 0, 1 -> squared: 1, 0, 1 -> mean: 2/3
        assert mse(y_true, y_pred) == pytest.approx(2 / 3, rel=1e-5)

    def test_rmse_is_sqrt_mse(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 2.5])
        assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(mse(y_true, y_pred)))

    def test_mae_known(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 2.0, 2.0])
        assert mae(y_true, y_pred) == pytest.approx(2 / 3, rel=1e-5)

    def test_r2_perfect(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert r2(y, y) == pytest.approx(1.0)

    def test_r2_baseline_predicts_mean(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        y_mean = np.full_like(y, y.mean())
        assert r2(y, y_mean) == pytest.approx(0.0, abs=1e-10)

    def test_r2_negative_for_terrible_model(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([3.0, 2.0, 1.0])   # reversed
        assert r2(y_true, y_pred) < 0

    def test_accuracy_all_correct(self):
        y = np.array([0, 1, 0, 1])
        assert accuracy(y, y) == pytest.approx(1.0)

    def test_accuracy_all_wrong(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        assert accuracy(y_true, y_pred) == pytest.approx(0.0)

    def test_accuracy_half(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])
        assert accuracy(y_true, y_pred) == pytest.approx(0.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
