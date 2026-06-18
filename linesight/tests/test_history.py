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




from linesight.utils.history import TrainingHistory


class TestTrainingHistory:

    def test_is_empty_default(self):
        h = TrainingHistory()
        assert h.is_empty() is True

    def test_is_empty_false_after_losses_added(self):
        h = TrainingHistory(losses=[0.5, 0.3, 0.2])
        assert h.is_empty() is False

    def test_weights_default_empty(self):
        h = TrainingHistory()
        assert h.weights == []

    def test_biases_default_empty(self):
        h = TrainingHistory()
        assert h.biases == []

    def test_converged_default_false(self):
        h = TrainingHistory()
        assert h.converged is False

    def test_epochs_run_default_zero(self):
        h = TrainingHistory()
        assert h.epochs_run == 0

    def test_fields_populated(self):
        w = [np.array([1.0, 2.0])]
        h = TrainingHistory(losses=[0.5], weights=w, biases=[0.1],
                            learning_rate=0.01, epochs_run=1, converged=True)
        assert h.losses == [0.5]
        assert h.biases == [0.1]
        assert h.converged is True
        assert h.learning_rate == 0.01


