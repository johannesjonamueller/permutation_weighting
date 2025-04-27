"""
Main estimator for permutation weighting.
"""

# Use relative imports to ensure proper module resolution
from .data_validation import check_data, check_eval_data, is_data_binary
from .data_factory import get_data_factory, get_binary_data_factory
from .trainer_factory import get_trainer_factory
from .evaluator import WeightsPassthrough, evaluator_factory


def PW(A, X, classifier='logit', estimand='ATE', classifier_params=None,
       estimand_params=None, eval_data=None, num_replicates=100,
       evaluator_names=None, batch_size=None):
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
        Classification method ('logit', 'boosting', 'sgd', 'mlp')
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
    batch_size : int, optional
        Size of mini-batches for training SGD and MLP models

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

    if evaluator_names is None:
        evaluator_names = ['mse', 'logloss']

    # Add batch_size to classifier_params if provided and classifier supports it
    if batch_size is not None and classifier in ['sgd', 'mlp']:
        classifier_params['permute_batch_size'] = batch_size

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

    # Get trainer factory - now pass A to determine if treatment is binary
    trainer_factory = get_trainer_factory(classifier, classifier_params)

    # Run replicates
    eval_list = []
    convergence_info = {'converged': True, 'iterations': 0}

    for _ in range(num_replicates):
        data = train_data_factory()
        edata = eval_data_factory()
        model = trainer_factory(data)

        # Capture convergence information if available
        if hasattr(model, 'convergence_info'):
            rep_convergence = getattr(model, 'convergence_info', {})
            # Update global convergence info
            convergence_info['converged'] = convergence_info['converged'] and rep_convergence.get('converged', True)
            convergence_info['iterations'] = max(convergence_info['iterations'], rep_convergence.get('iterations', 0))

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
        'batch_size': batch_size
    }
    results['convergence_info'] = convergence_info

    return results