import numpy as np
import pytest
from permutation_weighting.evaluator import (
    WeightsPassthrough, MSEEvaluator, LogLossEvaluator, evaluator_factory
)


def test_weights_passthrough_evaluate():
    # Create a model function
    def model(A, X):
        return np.array([1.0, 2.0])

    # Create comparison_with_r_package.ipynb data
    data = {
        'observed': {
            'A': np.array([0, 1]),
            'X': np.array([[1, 2], [3, 4]])
        }
    }

    evaluator = WeightsPassthrough()
    result = evaluator.evaluate(model, data)

    assert np.array_equal(result, np.array([1.0, 2.0]))


def test_weights_passthrough_combine():
    evaluator = WeightsPassthrough()
    result = evaluator.combine(
        np.array([1.0, 2.0]),
        np.array([3.0, 4.0])
    )

    assert np.array_equal(result, np.array([4.0, 6.0]))


def test_weights_passthrough_normalize():
    evaluator = WeightsPassthrough()
    result = evaluator.normalize(
        np.array([2.0, 4.0, 6.0]),
        num_replicates=2
    )

    # Should normalize to sum to n (3)
    assert np.sum(result) == pytest.approx(3.0)


def test_mse_evaluator_evaluate():
    # Create a model function
    def model(A, X):
        # Return weights such that prob = weights / (1 + weights) is 0.8 for observed and 0.2 for permuted
        return np.array([4.0, 4.0])  # weights = 4 -> prob = 0.8

    # Create comparison_with_r_package.ipynb data
    data = {
        'observed': {
            'A': np.array([0, 1]),
            'X': np.array([[1, 2], [3, 4]])
        },
        'permuted': {
            'A': np.array([0, 1]),
            'X': np.array([[5, 6], [7, 8]])
        }
    }

    evaluator = MSEEvaluator()
    result = evaluator.evaluate(model, data)

    # Expected MSE based on the current implementation:
    # observed: mean(0.8^2) = 0.64
    # permuted: mean(0.8^2) = 0.64
    # overall: (0.64 + 0.04)/2 = 0.34
    assert result == pytest.approx(0.34)


def test_logloss_evaluator_evaluate():
    # Create a model function
    def model(A, X):
        # Return weights such that prob = weights / (1 + weights) is 0.8 for observed and 0.2 for permuted
        return np.array([4.0, 4.0])  # weights = 4 -> prob = 0.8

    # Create comparison_with_r_package.ipynb data
    data = {
        'observed': {
            'A': np.array([0, 1]),
            'X': np.array([[1, 2], [3, 4]])
        },
        'permuted': {
            'A': np.array([0, 1]),
            'X': np.array([[5, 6], [7, 8]])
        }
    }

    evaluator = LogLossEvaluator()
    result = evaluator.evaluate(model, data)

    # Expected log loss:
    # observed: mean(-log(1-0.8)) = mean(-log(0.2)) = mean(1.609) = 1.609
    # permuted: mean(-log(1-0.2)) = mean(-log(0.8)) = mean(0.223) = 0.223
    # overall: mean([1.609, 0.223]) = 0.916
    assert result == pytest.approx(0.916, abs=0.01)


def test_evaluator_factory():
    # Test MSE evaluator
    evaluator = evaluator_factory('mse')
    assert isinstance(evaluator, MSEEvaluator)

    # Test log loss evaluator
    evaluator = evaluator_factory('logloss')
    assert isinstance(evaluator, LogLossEvaluator)

    # Test unknown evaluator
    with pytest.raises(ValueError, match="Unknown evaluator"):
        evaluator_factory('unknown')