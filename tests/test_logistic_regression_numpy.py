import numpy as np

from src.logistic_regression_numpy import LogisticRegressionNumpy


def test_numpy_logistic_regression_learns_linearly_separable_data() -> None:
    features = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    target = np.array([0, 0, 1, 1])

    model = LogisticRegressionNumpy(learning_rate=0.2, epochs=1_000).fit(features, target)

    assert np.array_equal(model.predict(features), target)
    assert model.loss_history_[-1] < model.loss_history_[0]
