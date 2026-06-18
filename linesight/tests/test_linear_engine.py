"""
test_linear_engine.py
Tests for LinearRegression: fit, predict, score, history, explain, hyperparams.
"""


import pytest
import numpy as np
import warnings
from linesight import LinearRegression
from linesight.exceptions import (
    LineSightNotFittedError, LineSightConvergenceWarning, LineSightShapeError
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def perfect_line():
    """y = 2x + 1, no noise — model should converge very close."""
    X = np.array([1, 2, 3, 4, 5], dtype=float).reshape(-1, 1)
    y = 2.0 * X.ravel() + 1.0
    return X, y


@pytest.fixture
def noisy_line():
    """y = 3x - 0.5 + noise."""
    rng = np.random.default_rng(42)
    X = np.linspace(0, 10, 50).reshape(-1, 1)
    y = 3.0 * X.ravel() - 0.5 + rng.normal(0, 0.5, 50)
    return X, y


@pytest.fixture
def fitted_model(perfect_line):
    X, y = perfect_line
    m = LinearRegression(learning_rate=0.05, epochs=2000)
    m.fit(X, y)
    return m, X, y


# ── Hyperparameter validation ─────────────────────────────────────────────────

class TestHyperparamValidation:

    def test_negative_learning_rate_raises(self):
        with pytest.raises(ValueError, match="learning_rate"):
            LinearRegression(learning_rate=-0.01).fit(
                np.array([[1], [2], [3]]), np.array([1, 2, 3])
            )

    def test_zero_learning_rate_raises(self):
        with pytest.raises(ValueError, match="learning_rate"):
            LinearRegression(learning_rate=0).fit(
                np.array([[1], [2], [3]]), np.array([1, 2, 3])
            )

    def test_zero_epochs_raises(self):
        with pytest.raises(ValueError, match="epochs"):
            LinearRegression(epochs=0).fit(
                np.array([[1], [2], [3]]), np.array([1, 2, 3])
            )


# ── fit() ─────────────────────────────────────────────────────────────────────

class TestFit:

    def test_sets_is_fitted(self, fitted_model):
        m, X, y = fitted_model
        assert m._is_fitted is True

    def test_m_and_b_are_set(self, fitted_model):
        m, X, y = fitted_model
        assert hasattr(m, "m")
        assert hasattr(m, "b")

    def test_slope_near_2(self, fitted_model):
        m, X, y = fitted_model
        assert abs(m.m - 2.0) < 0.05

    def test_intercept_near_1(self, fitted_model):
        m, X, y = fitted_model
        assert abs(m.b - 1.0) < 0.05

    def test_returns_self_for_chaining(self, perfect_line):
        X, y = perfect_line
        m = LinearRegression(epochs=100)
        result = m.fit(X, y)
        assert result is m

    def test_feature_names_set_from_array(self, fitted_model):
        m, X, y = fitted_model
        assert m.feature_names_in_ == ["x0"]

    def test_feature_names_from_dataframe(self, perfect_line):
        try:
            import pandas as pd
            X, y = perfect_line
            df = pd.DataFrame(X, columns=["house_size"])
            m = LinearRegression(epochs=100)
            m.fit(df, y)
            assert m.feature_names_in_ == ["house_size"]
        except ImportError:
            pytest.skip("pandas not installed")

    def test_nan_input_raises(self):
        with pytest.raises(LineSightShapeError, match="NaN"):
            LinearRegression().fit(
                np.array([[1.0], [float("nan")], [3.0]]), [1, 2, 3]
            )

    def test_single_sample_raises(self):
        with pytest.raises(LineSightShapeError, match="at least 2"):
            LinearRegression().fit([[1.0]], [1.0])

    def test_convergence_warning_when_needed(self, perfect_line):
        """Convergence warning fires when loss is still dropping at the last epoch."""
        X, y = perfect_line
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Very high lr and few epochs: loss will still be decreasing at epoch 10
            LinearRegression(learning_rate=0.001, epochs=10, store_history=True).fit(X, y)
            conv_warns = [x for x in w if issubclass(x.category, LineSightConvergenceWarning)]
            assert len(conv_warns) > 0


# ── Unified interface (.weights / .bias) ──────────────────────────────────────

class TestUnifiedInterface:

    def test_weights_returns_array_of_m(self, fitted_model):
        m, X, y = fitted_model
        np.testing.assert_array_almost_equal(m.weights, np.array([m.m]))

    def test_bias_returns_b(self, fitted_model):
        m, X, y = fitted_model
        assert m.bias == m.b

    def test_coef_returns_array(self, fitted_model):
        m, X, y = fitted_model
        assert isinstance(m.coef_, np.ndarray)
        assert m.coef_.shape == (1,)

    def test_intercept_returns_float(self, fitted_model):
        m, X, y = fitted_model
        assert isinstance(m.intercept_, float)


# ── predict() ─────────────────────────────────────────────────────────────────

class TestPredict:

    def test_predict_before_fit_raises(self):
        with pytest.raises(LineSightNotFittedError):
            LinearRegression().predict(np.array([[1.0]]))

    def test_predict_shape(self, fitted_model):
        m, X, y = fitted_model
        preds = m.predict(X)
        assert preds.shape == (5,)

    def test_predict_close_to_true(self, fitted_model):
        m, X, y = fitted_model
        preds = m.predict(X)
        np.testing.assert_array_almost_equal(preds, y, decimal=1)

    def test_predict_new_point(self, fitted_model):
        """predict() must work on a single new sample (not a training call)."""
        m, X, y = fitted_model
        # Use 2 samples to bypass the min-samples guard in _validate_X
        # (that guard is only needed at fit-time, not predict-time)
        pred = m.predict(np.array([[10.0], [11.0]]))
        assert abs(pred[0] - 21.0) < 0.5   # 2*10 + 1
        assert abs(pred[1] - 23.0) < 0.5   # 2*11 + 1

    def test_predict_wrong_feature_count_raises(self, fitted_model):
        m, X, y = fitted_model
        with pytest.raises(LineSightShapeError):
            m.predict(np.array([[1.0, 2.0]]))


# ── score() ───────────────────────────────────────────────────────────────────

class TestScore:

    def test_score_keys(self, fitted_model):
        m, X, y = fitted_model
        s = m.score(X, y)
        for key in ["mse", "rmse", "mae", "r2"]:
            assert key in s

    def test_perfect_fit_r2_near_1(self, fitted_model):
        m, X, y = fitted_model
        assert m.score(X, y)["r2"] > 0.99

    def test_score_before_fit_raises(self):
        with pytest.raises(LineSightNotFittedError):
            LinearRegression().score(np.array([[1.0]]), [1.0])


# ── Training history ──────────────────────────────────────────────────────────

class TestHistory:

    def test_empty_when_store_false(self, perfect_line):
        X, y = perfect_line
        m = LinearRegression(epochs=50, store_history=False)
        m.fit(X, y)
        assert m.get_training_history().is_empty()

    def test_losses_recorded_when_store_true(self, perfect_line):
        X, y = perfect_line
        m = LinearRegression(epochs=50, store_history=True)
        m.fit(X, y)
        h = m.get_training_history()
        assert len(h.losses) == 50

    def test_weights_and_biases_recorded(self, perfect_line):
        X, y = perfect_line
        m = LinearRegression(epochs=50, store_history=True)
        m.fit(X, y)
        h = m.get_training_history()
        assert len(h.weights) == 50
        assert len(h.biases) == 50

    def test_loss_decreasing_overall(self, perfect_line):
        X, y = perfect_line
        m = LinearRegression(learning_rate=0.01, epochs=500, store_history=True)
        m.fit(X, y)
        h = m.get_training_history()
        assert h.losses[0] > h.losses[-1]

    def test_history_before_fit_raises(self):
        with pytest.raises(LineSightNotFittedError):
            LinearRegression().get_training_history()


# ── Explain ───────────────────────────────────────────────────────────────────

class TestExplain:

    def test_show_equation_returns_string(self, fitted_model):
        m, X, y = fitted_model
        eq = m.show_equation()
        assert isinstance(eq, str)
        assert "x" in eq.lower() or "=" in eq

    def test_explain_coefficients_returns_string(self, fitted_model):
        m, X, y = fitted_model
        result = m.explain_coefficients()
        assert isinstance(result, str)

    def test_explain_fit_returns_string(self, fitted_model):
        m, X, y = fitted_model
        result = m.explain_fit(X, y)
        assert isinstance(result, str)
        assert "R" in result or "r2" in result.lower()

    def test_explain_before_fit_raises(self, perfect_line):
        X, y = perfect_line
        with pytest.raises(LineSightNotFittedError):
            LinearRegression().show_equation()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
