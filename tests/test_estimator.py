import numpy as np
import pytest
from permutation_weighting.estimator import PW


def test_pw_init():
    # Create simple data
    np.random.seed(42)
    n = 20
    X = np.random.normal(size=(n, 2))
    propensity = 1 / (1 + np.exp(-X[:, 0] + 0.5 * X[:, 1]))
    A = np.random.binomial(1, propensity, size=n)

    # Run PW with bootstrap=True to override default behavior
    result = PW(A, X, num_replicates=2, estimand_params={'bootstrap': True})

    # Check result structure
    assert 'weights' in result
    assert 'train' in result
    assert 'call' in result

    # Check weights
    assert len(result['weights']) == n
    assert np.all(result['weights'] >= 0)

    # Check training metrics
    assert 'MSEEvaluator' in result['train']
    assert 'LogLossEvaluator' in result['train']

    # Check call info
    assert result['call']['classifier'] == 'logit'
    assert result['call']['estimand'] == 'ATE'
    assert result['call']['num_replicates'] == 2


def test_pw_eval_data():
    # Create simple comparison_with_r_package.ipynb data
    np.random.seed(42)
    n = 20
    X = np.random.normal(size=(n, 2))
    propensity = 1 / (1 + np.exp(-X[:, 0] + 0.5 * X[:, 1]))
    A = np.random.binomial(1, propensity, size=n)

    # Create eval data
    eval_data = {
        'A': A.copy(),
        'X': X.copy()
    }

    # Run PW with eval data
    result = PW(A, X, eval_data=eval_data, num_replicates=2)

    # Check result structure
    assert 'eval' in result

    # Check eval metrics
    assert 'MSEEvaluator' in result['eval']
    assert 'LogLossEvaluator' in result['eval']


def test_pw_binary_crossproduct():
    # Create simple binary comparison_with_r_package.ipynb data
    np.random.seed(42)
    n = 20
    X = np.random.normal(size=(n, 2))
    A = np.array([0, 1] * (n // 2))

    # Run PW with binary data (should use crossproduct)
    result = PW(A, X)

    # Since using crossproduct, should only have 1 replicate regardless of num_replicates
    assert result['call']['num_replicates'] == 1

    # Check weights
    assert len(result['weights']) == n
    assert np.all(result['weights'] >= 0)


def test_pw_bootstrap():
    # Create simple binary comparison_with_r_package.ipynb data
    np.random.seed(42)
    n = 20
    X = np.random.normal(size=(n, 2))
    A = np.array([0, 1] * (n // 2))

    # Run PW with bootstrap=True to override crossproduct
    result = PW(A, X, estimand_params={'bootstrap': True}, num_replicates=2)

    # Should use specified num_replicates
    assert result['call']['num_replicates'] == 2

    # Check weights
    assert len(result['weights']) == n
    assert np.all(result['weights'] >= 0)


def test_pw_boosting():
    # Create simple comparison_with_r_package.ipynb data
    np.random.seed(42)
    n = 20
    X = np.random.normal(size=(n, 2))
    propensity = 1 / (1 + np.exp(-X[:, 0] + 0.5 * X[:, 1]))
    A = np.random.binomial(1, propensity, size=n)

    # Run PW with boosting
    result = PW(A, X, classifier='boosting', num_replicates=2)

    # Check classifier in call info
    assert result['call']['classifier'] == 'boosting'

    # Check weights
    assert len(result['weights']) == n
    assert np.all(result['weights'] >= 0)


def test_pw_att():
    # Create simple comparison_with_r_package.ipynb data
    np.random.seed(42)
    n = 20
    X = np.random.normal(size=(n, 2))
    propensity = 1 / (1 + np.exp(-X[:, 0] + 0.5 * X[:, 1]))
    A = np.random.binomial(1, propensity, size=n)

    # Ensure at least one treated unit
    A[0] = 1

    # Run PW with ATT estimand
    result = PW(A, X, estimand='ATT', num_replicates=2)

    # Check estimand in call info
    assert result['call']['estimand'] == 'ATT'

    # Check weights
    assert len(result['weights']) == n
    assert np.all(result['weights'] >= 0)


#New test for torch #TODO: Better understand the tests


def test_pw_with_torch_logistic():
    """Test PW with PyTorch logistic regression"""
    try:
        import torch
        torch_available = True
    except ImportError:
        torch_available = False

    if not torch_available:
        pytest.skip("PyTorch not available, skipping test")

    # Create simple data
    np.random.seed(42)
    n = 20
    X = np.random.normal(size=(n, 2))
    propensity = 1 / (1 + np.exp(-X[:, 0] + 0.5 * X[:, 1]))
    A = np.random.binomial(1, propensity, size=n)

    # Run PW with PyTorch logistic regression
    result = PW(A, X, classifier='logistic', use_torch=True,
                estimand_params={'bootstrap': True}, num_replicates=2)

    # Check result structure
    assert 'weights' in result
    assert 'train' in result
    assert 'call' in result

    # Check weights
    assert len(result['weights']) == n
    assert np.all(np.isfinite(result['weights']))  # No NaN or Inf
    assert np.all(result['weights'] >= 0)

    # Check call info
    assert result['call']['classifier'] == 'logistic'
    assert result['call']['use_torch'] is True
    assert result['call']['num_replicates'] == 2


def test_pw_with_torch_mlp():
    """Test PW with PyTorch MLP"""
    try:
        import torch
        torch_available = True
    except ImportError:
        torch_available = False

    if not torch_available:
        pytest.skip("PyTorch not available, skipping test")

    # Create simple data
    np.random.seed(42)
    n = 20
    X = np.random.normal(size=(n, 2))
    propensity = 1 / (1 + np.exp(-X[:, 0] + 0.5 * X[:, 1]))
    A = np.random.binomial(1, propensity, size=n)

    # Custom MLP parameters
    params = {'hidden_dims': [8, 4], 'epochs': 5, 'batch_size': 4}

    # Run PW with PyTorch MLP
    result = PW(A, X, classifier='mlp', use_torch=True,
                classifier_params=params,
                estimand_params={'bootstrap': True}, num_replicates=2)

    # Check result
    assert 'weights' in result
    assert len(result['weights']) == n
    assert np.all(np.isfinite(result['weights']))
    assert np.all(result['weights'] >= 0)

    # Check call info
    assert result['call']['classifier'] == 'mlp'
    assert result['call']['use_torch'] is True
    assert result['call']['classifier_params'] == params


def test_pw_with_torch_custom():
    """Test PW with custom PyTorch configuration"""
    try:
        import torch
        torch_available = True
    except ImportError:
        torch_available = False

    if not torch_available:
        pytest.skip("PyTorch not available, skipping test")

    # Create simple data
    np.random.seed(42)
    n = 20
    X = np.random.normal(size=(n, 2))
    propensity = 1 / (1 + np.exp(-X[:, 0] + 0.5 * X[:, 1]))
    A = np.random.binomial(1, propensity, size=n)

    # Custom parameters with model type specification
    params = {
        'model_type': 'resnet',
        'hidden_dim': 8,
        'num_blocks': 1,
        'epochs': 5,
        'batch_size': 4
    }

    # Run PW with custom PyTorch configuration
    result = PW(A, X, classifier='torch_custom', use_torch=True,
                classifier_params=params,
                estimand_params={'bootstrap': True}, num_replicates=2)

    # Check result
    assert 'weights' in result
    assert len(result['weights']) == n
    assert np.all(np.isfinite(result['weights']))
    assert np.all(result['weights'] >= 0)

    # Check call info
    assert result['call']['classifier'] == 'torch_custom'
    assert result['call']['use_torch'] is True
    assert result['call']['classifier_params'] == params


def test_pw_torch_not_available():
    """Test that PW fails gracefully when PyTorch is not available"""
    # Create simple data
    np.random.seed(42)
    n = 20
    X = np.random.normal(size=(n, 2))
    A = np.random.binomial(1, 0.5, size=n)

    # Monkeypatch TORCH_AVAILABLE to False
    import permutation_weighting.estimator as estimator
    original_value = getattr(estimator, 'TORCH_AVAILABLE', True)
    setattr(estimator, 'TORCH_AVAILABLE', False)

    try:
        # Should raise ImportError when PyTorch not available
        with pytest.raises(ImportError, match="PyTorch is not available"):
            PW(A, X, classifier='logistic', use_torch=True,
               estimand_params={'bootstrap': True}, num_replicates=2)
    finally:
        # Restore original value
      setattr(estimator, 'TORCH_AVAILABLE', original_value)


def test_neural_network_replicates():
    """Test that neural network models use fewer replicates automatically"""
    np.random.seed(42)
    n = 100
    X = np.random.normal(size=(n, 2))
    propensity = 1 / (1 + np.exp(-X[:, 0] + 0.5 * X[:, 1]))
    A = np.random.binomial(1, propensity, size=n)

    # Create a spy to monitor the number of model trainings
    train_count = [0]  # Use list to allow modification in nested function

    # Original trainer factory
    from permutation_weighting.models.sgd_trainer_factory import neural_net_factory
    original_factory = neural_net_factory

    # Create a wrapper to count calls
    def counting_factory(params=None):
        original = original_factory(params)

        def wrapper(data):
            train_count[0] += 1
            return original(data)

        return wrapper

    # Patch the factory method
    import permutation_weighting.estimator as estimator
    original_get_trainer = estimator.get_trainer_factory
    estimator.neural_net_factory = counting_factory

    try:
        # Run PW with neural network and 10 requested replicates
        result = PW(
            A=A,
            X=X,
            classifier='neural_net',
            use_sgd=True,
            num_replicates=10
        )

        # Should only train once despite requesting 10 replicates
        assert train_count[0] == 1, f"Expected 1 training run, got {train_count[0]}"
        assert result['call']['num_replicates'] == 10, "Original replicate count should be preserved in call info"

        # Compare with standard logistic that should use all replicates
        train_count[0] = 0
        result_logit = PW(
            A=A,
            X=X,
            classifier='logit',
            num_replicates=5
        )

        assert train_count[0] == 0, "Logit shouldn't use the counting wrapper"

    finally:
        # Restore original function
        estimator.neural_net_factory = original_factory

    print("Neural network replicate reduction test passed!")


def test_convergence_info_communication():
    """Test that convergence information is properly communicated"""
    # Generate simple data
    np.random.seed(42)
    n = 100
    X = np.random.normal(size=(n, 2))
    A = np.random.binomial(1, 0.5, size=n)

    # Create a custom trainer that adds convergence info
    def custom_trainer_factory(data):
        # Simple logistic regression
        from sklearn.linear_model import LogisticRegression

        # Extract data
        X_train = np.column_stack([
            data['observed']['A'],
            data['observed']['X']
        ])
        y_train = np.zeros(len(data['observed']['A']))

        X_perm = np.column_stack([
            data['permuted']['A'],
            data['permuted']['X']
        ])
        y_perm = np.ones(len(data['permuted']['A']))

        # Combined dataset
        X_combined = np.vstack([X_train, X_perm])
        y_combined = np.concatenate([y_train, y_perm])

        # Train model
        model = LogisticRegression()
        model.fit(X_combined, y_combined)

        # Create weight function
        def weight_func(A_new, X_new):
            # Predict on new data
            X_eval = np.column_stack([A_new, X_new])
            probs = model.predict_proba(X_eval)[:, 1]
            weights = probs / (1 - probs)
            return weights

        # Add convergence info
        weight_func.convergence_info = {
            'converged': True,
            'iterations': 42,
            'best_loss': 0.123,
            'final_loss': 0.234,
            'custom_metric': 0.987
        }

        return weight_func

    # Patch get_trainer_factory to return our custom factory
    import permutation_weighting.estimator as estimator
    original_get_trainer = estimator.get_trainer_factory

    def mock_get_trainer(classifier, params=None):
        return custom_trainer_factory

    estimator.get_trainer_factory = mock_get_trainer

    try:
        # Run PW with our custom trainer
        result = PW(
            A=A,
            X=X,
            classifier='custom',  # Name doesn't matter, we've patched the factory
            num_replicates=3,
            estimand_params={'bootstrap': True}  # Add this to force multiple replicates
        )

        # Check if convergence info was captured and aggregated
        assert 'convergence_info' in result, "Convergence info should be present"
        assert 'iterations' in result['convergence_info'], "Should capture iterations"
        assert result['convergence_info']['iterations'] == 42, "Should use max iterations from factories"

        assert 'details' in result['convergence_info'], "Should capture detailed convergence info"
        for detail in result['convergence_info']['details']:
            assert 'final_loss' in detail, "Each detail record should include final_loss"


        # Check if custom metric was preserved
        assert result['convergence_info']['details'][0]['custom_metric'] == 0.987, "Should preserve custom metrics"

        # Check if losses were captured
        assert 'losses' in result['convergence_info'], "Should capture loss values"
        assert len(result['convergence_info']['losses']) == 3, "Should have 3 loss values"
        assert result['convergence_info']['losses'][0] == 0.234, "Should capture final_loss values"

        # Check if best_loss was captured
        assert 'best_loss' in result['convergence_info'], "Should capture best_loss"
        assert result['convergence_info']['best_loss'] == 0.123, "Should use min best_loss from factories"

    finally:
        # Restore original function
        estimator.get_trainer_factory = original_get_trainer

    print("Convergence info communication test passed!")