import numpy as np
import pandas as pd
import pytest
from permutation_weighting.trainer_factory import (
    get_trainer_factory, construct_df, construct_eval_df,
    logit_factory, boosting_factory, sgd_factory, mlp_factory
)


def test_get_trainer_factory_unknown_classifier():
    with pytest.raises(ValueError, match="Unknown classifier"):
        get_trainer_factory('unknown')


def test_construct_df():
    # Create test data
    data = {
        'permuted': {
            'C': 1,
            'A': np.array([0, 1]),
            'X': np.array([[1, 2], [3, 4]])
        },
        'observed': {
            'C': 0,
            'A': np.array([0, 1]),
            'X': np.array([[5, 6], [7, 8]])
        }
    }

    df = construct_df(data)

    # Check DataFrame structure
    assert isinstance(df, pd.DataFrame)
    assert 'C' in df.columns
    assert 'A' in df.columns
    assert 'X0' in df.columns
    assert 'X1' in df.columns
    assert 'A_X0' in df.columns
    assert 'A_X1' in df.columns

    # Check DataFrame values
    assert len(df) == 4  # 2 permuted + 2 observed
    assert np.array_equal(df['C'].values, [1, 1, 0, 0])
    assert np.array_equal(df['A'].values, [0, 1, 0, 1])
    assert np.array_equal(df['X0'].values, [1, 3, 5, 7])
    assert np.array_equal(df['X1'].values, [2, 4, 6, 8])


def test_construct_eval_df():
    A = np.array([0, 1])
    X = np.array([[1, 2], [3, 4]])

    df = construct_eval_df(A, X)

    # Check DataFrame structure
    assert isinstance(df, pd.DataFrame)
    assert 'A' in df.columns
    assert 'X0' in df.columns
    assert 'X1' in df.columns
    assert 'A_X0' in df.columns
    assert 'A_X1' in df.columns

    # Check DataFrame values
    assert len(df) == 2
    assert np.array_equal(df['A'].values, [0, 1])
    assert np.array_equal(df['X0'].values, [1, 3])
    assert np.array_equal(df['X1'].values, [2, 4])
    assert np.array_equal(df['A_X0'].values, [0, 3])  # 0*1, 1*3
    assert np.array_equal(df['A_X1'].values, [0, 4])  # 0*2, 1*4


def test_logit_factory():
    # Create test data
    data = {
        'permuted': {
            'C': 1,
            'A': np.array([0, 1]),
            'X': np.array([[1, 2], [3, 4]])
        },
        'observed': {
            'C': 0,
            'A': np.array([0, 1]),
            'X': np.array([[5, 6], [7, 8]])
        }
    }

    # Get trainer
    trainer = logit_factory()
    model = trainer(data)

    # Test weight computation
    A = np.array([0, 1])
    X = np.array([[1, 2], [3, 4]])

    weights = model(A, X)

    # Check weights
    assert len(weights) == 2
    assert np.all(weights >= 0)  # Weights should be non-negative


def test_boosting_factory():
    # Create test data
    data = {
        'permuted': {
            'C': 1,
            'A': np.array([0, 1]),
            'X': np.array([[1, 2], [3, 4]])
        },
        'observed': {
            'C': 0,
            'A': np.array([0, 1]),
            'X': np.array([[5, 6], [7, 8]])
        }
    }

    # Get trainer
    trainer = boosting_factory()
    model = trainer(data)

    # Test weight computation
    A = np.array([0, 1])
    X = np.array([[1, 2], [3, 4]])

    weights = model(A, X)

    # Check weights
    assert len(weights) == 2
    assert np.all(weights >= 0)  # Weights should be non-negative


def test_sgd_factory():
    # Create test data with more samples to support validation split
    data = {
        'permuted': {
            'C': 1,
            'A': np.array([0, 0, 1, 1, 0, 1]),
            'X': np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
        },
        'observed': {
            'C': 0,
            'A': np.array([0, 0, 1, 1, 0, 1]),
            'X': np.array([[13, 14], [15, 16], [17, 18], [19, 20], [21, 22], [23, 24]])
        }
    }

    # Get trainer with validation disabled or smaller validation fraction
    trainer = sgd_factory({'early_stopping': False})
    model = trainer(data)

    # Test weight computation
    A = np.array([0, 1])
    X = np.array([[1, 2], [3, 4]])

    weights = model(A, X)

    # Check weights
    assert len(weights) == 2
    assert np.all(weights >= 0)  # Weights should be non-negative

    # Check convergence info
    assert hasattr(model, 'convergence_info')
    assert 'converged' in model.convergence_info
    assert 'iterations' in model.convergence_info


def test_mlp_factory():
    # Create test data with more samples to support validation split
    data = {
        'permuted': {
            'C': 1,
            'A': np.array([0, 0, 1, 1, 0, 1]),
            'X': np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
        },
        'observed': {
            'C': 0,
            'A': np.array([0, 0, 1, 1, 0, 1]),
            'X': np.array([[13, 14], [15, 16], [17, 18], [19, 20], [21, 22], [23, 24]])
        }
    }

    # Get trainer with validation disabled or smaller validation fraction
    trainer = mlp_factory({'early_stopping': False})
    model = trainer(data)


    # Test weight computation
    A = np.array([0, 1])
    X = np.array([[1, 2], [3, 4]])

    weights = model(A, X)

    # Check weights
    assert len(weights) == 2
    assert np.all(weights >= 0)  # Weights should be non-negative

    # Check convergence info
    assert hasattr(model, 'convergence_info')
    assert 'converged' in model.convergence_info
    assert 'iterations' in model.convergence_info