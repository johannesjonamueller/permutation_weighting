"""
Main estimator for permutation weighting.
"""

import numpy as np
# Use relative imports to ensure proper module resolution
from .data.data_validation import check_data, check_eval_data, is_data_binary
from .data.data_factory import get_data_factory, get_binary_data_factory
from .models.trainer_factory import get_trainer_factory
from .models.sgd_trainer_factory import sgd_logit_factory, neural_net_factory, minibatch_permute_trainer_factory
from .evaluator import WeightsPassthrough, evaluator_factory

# Import PyTorch trainers conditionally to avoid errors if PyTorch is not installed
try:
    from .models.torch_trainer_factory import (
        torch_trainer_factory, minibatch_permute_torch_factory, logistic_torch_factory,
        mlp_torch_factory, resnet_torch_factory
    )
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def PW(A, X, classifier='logit', estimand='ATE', classifier_params=None,
       estimand_params=None, eval_data=None,  num_replicates=100,
       evaluator_names=None, use_sgd=False, use_torch=False, batch_size=None):
    """
    Estimates non-parametric balancing weights for observational causal inference
    using permutation weighting.

    Parameters
    ----------
    A : array-like
        Treatment variable (binary or continuous)
    X : array-like
        Covariate matrix
    classifier : str, default='logit'
        Classification method ('logit', 'boosting', 'sgd_logit', 'neural_net')
    estimand : str, default='ATE'
        Target estimand ('ATE' or 'ATT')
    classifier_params : dict, optional
        Parameters for the classifier
    estimand_params : dict, optional
        Parameters for the estimand
    eval_data : dict, optional
        Evaluation data with 'A' and 'X' keys
    num_replicates : int, default=100
        Number of replicates to use
    evaluator_names : list, optional
        Names of evaluators to use
    use_sgd : bool, default=False
            Whether to use SGD-based training
    use_torch : bool, default=False
        Whether to use PyTorch-based training
    batch_size : int, optional
        Size of minibatches for SGD training

    Returns
    -------
    dict
        Dictionary containing weights and evaluation metrics
    """
    # Validate inputs
    A, X = check_data(A, X)

    if classifier_params is None:
        classifier_params = {}

    if estimand_params is None:
        estimand_params = {}

    # Make bootstrap=True the default for SGD methods
    if use_sgd and 'bootstrap' not in estimand_params:
        estimand_params['bootstrap'] = True

    if evaluator_names is None:
        evaluator_names = ['mse', 'logloss']

    # Check if data is binary
    is_binary = is_data_binary(A)
    use_crossproduct = is_binary

    # Check if bootstrap is requested
    if 'bootstrap' in estimand_params and estimand_params['bootstrap']:
        use_crossproduct = False

    # Select appropriate data factory
    if use_crossproduct:
        train_data_factory = get_binary_data_factory(A, X, estimand, estimand_params)
        if num_replicates > 1:
            print("Warning: Disabling replicates on binary data. "
                  "Override this behavior by setting `bootstrap=True` in `estimand_params`.")
            num_replicates = 1
    else:
        train_data_factory = get_data_factory(A, X, estimand, estimand_params)

    # Set up evaluation data factory
    has_eval_data = eval_data is not None
    if has_eval_data:
        eval_data = check_eval_data(eval_data)
        eval_data_factory = get_data_factory(eval_data['A'], eval_data['X'], estimand)
    else:
        eval_data_factory = get_data_factory(A, X, estimand)

    # Create evaluators
    evaluators = [WeightsPassthrough()]
    for evaluator_name in evaluator_names:
        evaluators.append(evaluator_factory(evaluator_name))


    # Get trainer factory
    if use_torch:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. Install with 'pip install torch'")

        if batch_size is not None:
            # Use the batch-then-permute approach
            trainer_factory = minibatch_permute_torch_factory(
                classifier if classifier != 'torch_custom' else classifier_params.get('model_type', 'logistic'),
                classifier_params
            )
        elif classifier == 'logistic':
            trainer_factory = logistic_torch_factory(classifier_params)
        elif classifier == 'mlp':
            trainer_factory = mlp_torch_factory(classifier_params)
        elif classifier == 'resnet':
            trainer_factory = resnet_torch_factory(classifier_params)
        elif classifier == 'torch_custom':
            trainer_factory = torch_trainer_factory(classifier_params.get('model_type', 'logistic'), classifier_params)
        else:
            raise ValueError(f"Unknown PyTorch classifier: {classifier}")
    elif use_sgd:
        if batch_size is not None:
            # Use the batch-then-permute approach
            trainer_factory = minibatch_permute_trainer_factory(classifier, classifier_params, batch_size)
        elif classifier == 'logit':
            trainer_factory = sgd_logit_factory(classifier_params)
        elif classifier == 'neural_net':
            trainer_factory = neural_net_factory(classifier_params)
        else:
            raise ValueError(f"Unknown SGD classifier: {classifier}")
    else:
        trainer_factory = get_trainer_factory(classifier, classifier_params)

    # Run replicates
    eval_list = []
    convergence_info = {'converged': True, 'iterations': 0}

    effective_replicates = 1 if (use_torch or classifier in ['neural_net', 'mlp', 'resnet']) else num_replicates
    if effective_replicates < num_replicates:
        print(f"Using {effective_replicates} replicate(s) for {classifier} instead of {num_replicates}")

    for _ in range(effective_replicates):
        data = train_data_factory()
        edata = eval_data_factory()
        model = trainer_factory(data)

        # Capture and store more detailed convergence information
        if hasattr(model, 'convergence_info'):
            rep_convergence = getattr(model, 'convergence_info', {})

            # Initialize detailed convergence tracking if not present
            if 'details' not in convergence_info:
                convergence_info['details'] = []

            # Store this replicate's convergence info
            convergence_info['details'].append(rep_convergence)

            # Update global convergence info
            convergence_info['converged'] = convergence_info['converged'] and rep_convergence.get('converged', True)
            convergence_info['iterations'] = max(convergence_info['iterations'], rep_convergence.get('iterations', 0))

            # Track losses if available
            if 'final_loss' in rep_convergence:
                if 'losses' not in convergence_info:
                    convergence_info['losses'] = []
                convergence_info['losses'].append(rep_convergence['final_loss'])

            # Track best loss across all replicates
            if 'best_loss' in rep_convergence:
                if 'best_loss' not in convergence_info or rep_convergence['best_loss'] < convergence_info['best_loss']:
                    convergence_info['best_loss'] = rep_convergence['best_loss']

        ev = {"train": {}, "eval": {}}
        for evaluator in evaluators:
            class_name = evaluator.__class__.__name__
            ev["train"][class_name] = evaluator.evaluate(model, data)
            if has_eval_data:
                ev["eval"][class_name] = evaluator.evaluate(model, edata)

        eval_list.append(ev)

    # Aggregate results
    results = {"train": {}, "eval": {}}
    for evaluator in evaluators:
        class_name = evaluator.__class__.__name__
        eval_output = [e["train"][class_name] for e in eval_list]
        agg_result = evaluator.combine(*eval_output)
        results["train"][class_name] = evaluator.normalize(agg_result, num_replicates=num_replicates)

        if has_eval_data:
            eval_output = [e["eval"][class_name] for e in eval_list]
            agg_result = evaluator.combine(*eval_output)
            results["eval"][class_name] = evaluator.normalize(agg_result, num_replicates=num_replicates)

    # Set weights
    results['weights'] = results['train']['WeightsPassthrough']
    del results['train']['WeightsPassthrough']

    # Save call information
    results['call'] = {
        'A': A,
        'X': X,
        'classifier': classifier,
        'estimand': estimand,
        'classifier_params': classifier_params,
        'estimand_params': estimand_params,
        'num_replicates': num_replicates,
        'use_sgd': use_sgd,
        'use_torch': use_torch,
        'batch_size': batch_size
    }

    # Make sure convergence_info always has a 'details' key before returning
    if 'details' not in convergence_info:
        convergence_info['details'] = [{
            'iterations': convergence_info.get('iterations', 0),
            'converged': convergence_info.get('converged', True)
        }]

    results['convergence_info'] = convergence_info

    return results