import numpy as np
import pytest
from permutation_weighting.data.data_validation import (
    check_data, check_data_for_att, check_eval_data, is_data_binary
)


def test_check_data_valid():
    # Valid inputs should pass without errors
    A = np.array([0, 1, 0, 1])
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

    A_out, X_out = check_data(A, X)

    assert np.array_equal(A, A_out)
    assert np.array_equal(X, X_out)


def test_check_data_invalid_dim():
    # A should be 1D
    A = np.array([[0, 1], [0, 1]])
    X = np.array([[1, 2], [3, 4]])

    with pytest.raises(ValueError, match="A must be a 1-dimensional array"):
        check_data(A, X)

    # X should be 2D
    A = np.array([0, 1])
    X = np.array([1, 2])

    with pytest.raises(ValueError, match="X must be a 2-dimensional array"):
        check_data(A, X)


def test_check_data_non_numeric():
    # A should be numeric
    A = np.array(['a', 'b'])
    X = np.array([[1, 2], [3, 4]])

    with pytest.raises(ValueError, match="A must contain numeric values"):
        check_data(A, X)

    # X should be numeric
    A = np.array([0, 1])
    X = np.array([['a', 'b'], ['c', 'd']])

    with pytest.raises(ValueError, match="X must contain numeric values"):
        check_data(A, X)


def test_check_data_mismatched_length():
    # A and X should have same number of observations
    A = np.array([0, 1, 0])
    X = np.array([[1, 2], [3, 4]])

    with pytest.raises(ValueError, match="A and X must have the same number of observations"):
        check_data(A, X)


def test_check_data_for_att_valid():
    # Valid inputs for ATT
    A = np.array([0, 1])
    X = np.array([[1, 2], [3, 4]])

    A_out, X_out = check_data_for_att(A, X)

    assert np.array_equal(A, A_out)
    assert np.array_equal(X, X_out)


def test_check_data_for_att_not_binary():
    # A should have exactly 2 unique values for ATT
    A = np.array([0, 1, 2])
    X = np.array([[1, 2], [3, 4], [5, 6]])

    with pytest.raises(ValueError, match="A can only have two unique values for the ATT estimand"):
        check_data_for_att(A, X)


def test_check_data_for_att_no_treated():
    # A should have at least one 1 for ATT
    A = np.array([0, 0, 0])
    X = np.array([[1, 2], [3, 4], [5, 6]])

    with pytest.raises(ValueError, match="A must take the value of one at least once for the ATT"):
        check_data_for_att(A, X)


def test_check_eval_data_valid():
    # Valid eval data
    eval_data = {
        'A': np.array([0, 1]),
        'X': np.array([[1, 2], [3, 4]])
    }

    eval_data_out = check_eval_data(eval_data)

    assert eval_data is eval_data_out


def test_check_eval_data_not_dict():
    # eval_data should be a dict
    eval_data = np.array([0, 1])

    with pytest.raises(ValueError, match="eval_data must be a dictionary"):
        check_eval_data(eval_data)


def test_check_eval_data_missing_keys():
    # eval_data should have A and X keys
    eval_data = {'A': np.array([0, 1])}

    with pytest.raises(ValueError, match="eval_data must contain 'A' and 'X' keys"):
        check_eval_data(eval_data)


def test_is_data_binary():
    # Binary data
    A = np.array([0, 1, 0, 1])
    assert is_data_binary(A) is True

    # Non-binary data
    A = np.array([0, 1, 2])
    assert is_data_binary(A) is False

    # Single value (not binary)
    A = np.array([0, 0, 0])
    assert is_data_binary(A) is False