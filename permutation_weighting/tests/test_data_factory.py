import numpy as np
import pytest

from permutation_weighting.data.data_factory import (
    get_data_factory, get_binary_data_factory,
    ate_factory, att_factory, binary_ate_factory, binary_att_factory
)


def test_get_data_factory_unknown_estimand():
    A = np.array([0, 1])
    X = np.array([[1, 2], [3, 4]])

    with pytest.raises(ValueError, match="Unknown estimand"):
        get_data_factory(A, X, 'UNKNOWN')


def test_get_binary_data_factory_unknown_estimand():
    A = np.array([0, 1])
    X = np.array([[1, 2], [3, 4]])

    with pytest.raises(ValueError, match="Unknown estimand"):
        get_binary_data_factory(A, X, 'UNKNOWN')


def test_ate_factory():
    A = np.array([0, 0, 1, 1])
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

    # Set seed for reproducibility
    np.random.seed(42)

    factory = ate_factory(A, X, {'bootstrap': True})
    data = factory()

    # Check data structure
    assert 'permuted' in data
    assert 'observed' in data
    assert 'C' in data['permuted']
    assert 'A' in data['permuted']
    assert 'X' in data['permuted']
    assert 'C' in data['observed']
    assert 'A' in data['observed']
    assert 'X' in data['observed']

    # Check data values
    assert data['permuted']['C'] == 1
    assert data['observed']['C'] == 0
    assert len(data['permuted']['A']) == len(A)
    assert len(data['observed']['A']) == len(A)
    assert data['permuted']['X'].shape == X.shape
    assert data['observed']['X'].shape == X.shape


def test_att_factory():
    A = np.array([0, 0, 1, 1])
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

    # Set seed for reproducibility
    np.random.seed(42)

    factory = att_factory(A, X, {'bootstrap': True})
    data = factory()

    # Check data structure
    assert 'permuted' in data
    assert 'observed' in data
    assert 'C' in data['permuted']
    assert 'A' in data['permuted']
    assert 'X' in data['permuted']
    assert 'C' in data['observed']
    assert 'A' in data['observed']
    assert 'X' in data['observed']

    # Check data values
    assert data['permuted']['C'] == 1
    assert data['observed']['C'] == 0
    assert len(data['permuted']['A']) == len(A)
    assert len(data['observed']['A']) == len(A)
    assert data['permuted']['X'].shape == X.shape
    assert data['observed']['X'].shape == X.shape

    # Check that permuted X comes from treated units
    assert np.all(np.isin(data['permuted']['X'], X[A == 1]))


def test_binary_ate_factory():
    A = np.array([0, 0, 1, 1])
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

    factory = binary_ate_factory(A, X)
    data = factory()

    # Check data structure
    assert 'permuted' in data
    assert 'observed' in data

    # Check permuted data
    assert data['permuted']['C'] == 1
    assert len(data['permuted']['A']) == 2 * len(A)  # Should have 2 * n entries (for each treatment value)
    assert data['permuted']['X'].shape == (2 * len(A), X.shape[1])  # Should have 2 * n rows

    # Check observed data
    assert data['observed']['C'] == 0
    assert np.array_equal(data['observed']['A'], A)
    assert np.array_equal(data['observed']['X'], X)


def test_binary_att_factory():
    A = np.array([0, 0, 1, 1])
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

    factory = binary_att_factory(A, X)
    data = factory()

    # Check data structure
    assert 'permuted' in data
    assert 'observed' in data

    # Check permuted data
    assert data['permuted']['C'] == 1
    assert len(data['permuted']['A']) == np.sum(A == 1)  # Should have as many entries as treated units
    assert np.all(data['permuted']['A'] == 1)  # All units should be treated
    assert data['permuted']['X'].shape == (np.sum(A == 1), X.shape[1])  # Should have n_treated rows

    # Check observed data
    assert data['observed']['C'] == 0
    assert np.array_equal(data['observed']['A'], A)
    assert np.array_equal(data['observed']['X'], X)