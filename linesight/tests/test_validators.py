"""
test_validators.py
Tests for input validation, error messages, and edge cases in _validate_X / _validate_Xy.
"""


import pytest
import numpy as np
import warnings
from linesight.utils.validators import _validate_X, _validate_Xy, _validate_binary_y
from linesight.exceptions import LineSightShapeError, LineSightDataWarning


# ── _validate_X ───────────────────────────────────────────────────────────────

class TestValidateX:

    def test_1d_input_reshaped_to_2d(self):
        X = _validate_X([1, 2, 3])
        assert X.shape == (3, 1)

    def test_2d_input_unchanged(self):
        X = _validate_X([[1, 2], [3, 4], [5, 6]])
        assert X.shape == (3, 2)

    def test_pandas_series_converted(self):
        try:
            import pandas as pd
            s = pd.Series([1.0, 2.0, 3.0])
            X = _validate_X(s)
            assert isinstance(X, np.ndarray)
            assert X.shape == (3, 1)
        except ImportError:
            pytest.skip("pandas not installed")

    def test_pandas_dataframe_converted(self):
        try:
            import pandas as pd
            df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
            X = _validate_X(df)
            assert X.shape == (3, 2)
        except ImportError:
            pytest.skip("pandas not installed")

    def test_scalar_raises(self):
        with pytest.raises(LineSightShapeError, match="scalar"):
            _validate_X(np.array(5.0))

    def test_empty_raises(self):
        with pytest.raises(LineSightShapeError):
            _validate_X([])

    def test_single_sample_raises(self):
        """Min-samples check lives in _validate_Xy (fit-time), not _validate_X (predict-time)."""
        with pytest.raises(LineSightShapeError, match="at least 2"):
            _validate_Xy([[1.0, 2.0]], [1.0])

    def test_nan_raises(self):
        with pytest.raises(LineSightShapeError, match="NaN"):
            _validate_X([[1, float("nan")], [3, 4], [5, 6]])

    def test_inf_raises(self):
        with pytest.raises(LineSightShapeError, match="Inf"):
            _validate_X([[1, float("inf")], [3, 4], [5, 6]])

    def test_feature_count_mismatch_raises(self):
        with pytest.raises(LineSightShapeError, match="feature"):
            _validate_X([[1, 2], [3, 4]], expected_features=3)

    def test_feature_count_match_passes(self):
        X = _validate_X([[1, 2], [3, 4]], expected_features=2)
        assert X.shape[1] == 2

    def test_float_coercion(self):
        X = _validate_X([[1, 2], [3, 4]])
        assert X.dtype == float


# ── _validate_Xy ──────────────────────────────────────────────────────────────

class TestValidateXy:

    def test_matching_shapes_pass(self):
        X, y = _validate_Xy([[1], [2], [3]], [1, 2, 3])
        assert X.shape == (3, 1)
        assert y.shape == (3,)

    def test_mismatched_shapes_raise(self):
        with pytest.raises(LineSightShapeError):
            _validate_Xy([[1], [2], [3]], [1, 2])

    def test_nan_in_y_raises(self):
        with pytest.raises(LineSightShapeError, match="NaN"):
            _validate_Xy([[1], [2], [3]], [1, float("nan"), 3])

    def test_inf_in_y_raises(self):
        with pytest.raises(LineSightShapeError, match="Inf"):
            _validate_Xy([[1], [2], [3]], [1, float("inf"), 3])

    def test_zero_variance_warning(self):
        X = np.array([[1, 5], [2, 5], [3, 5], [4, 5], [5, 5]], dtype=float)
        y = np.array([1, 2, 3, 4, 5], dtype=float)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _validate_Xy(X, y)
            assert any("zero variance" in str(warning.message).lower() for warning in w)

    def test_y_raveled(self):
        _, y = _validate_Xy([[1], [2], [3]], [[1], [2], [3]])
        assert y.ndim == 1


# ── _validate_binary_y ────────────────────────────────────────────────────────

class TestValidateBinaryY:

    def test_valid_0_1_passes(self):
        _validate_binary_y(np.array([0, 1, 0, 1]))

    def test_valid_float_0_1_passes(self):
        _validate_binary_y(np.array([0.0, 1.0, 0.0]))

    def test_label_2_raises(self):
        with pytest.raises(LineSightShapeError, match="0s and 1s"):
            _validate_binary_y(np.array([0, 1, 2]))

    def test_negative_label_raises(self):
        with pytest.raises(LineSightShapeError):
            _validate_binary_y(np.array([-1, 1]))

    def test_string_labels_raise(self):
        with pytest.raises(Exception):
            _validate_binary_y(np.array(["cat", "dog"]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
