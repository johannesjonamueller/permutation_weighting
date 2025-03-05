"""
Evaluators for permutation weighting.
"""

import numpy as np
from abc import ABC, abstractmethod


class Evaluator(ABC):
    """
    Base class for evaluators
    """

    @abstractmethod
    def evaluate(self, model, data):
        """Evaluate a model on data"""
        pass

    @abstractmethod
    def combine(self, *results):
        """Combine multiple evaluation results"""
        pass

    @abstractmethod
    def normalize(self, combined_result, num_replicates, **kwargs):
        """Normalize combined results"""
        pass


class WeightsPassthrough(Evaluator):
    """
    Evaluator that simply passes through the weights
    """

    def evaluate(self, model, data):
        """
        Evaluate a model by computing weights

        Parameters
        ----------
        model : function
            Weight function
        data : dict
            Data dictionary

        Returns
        -------
        numpy.ndarray
            Computed weights
        """
        return model(data['observed']['A'], data['observed']['X'])

    def combine(self, *results):
        """
        Combine multiple weight arrays

        Parameters
        ----------
        *results : numpy.ndarray
            Weight arrays

        Returns
        -------
        numpy.ndarray
            Sum of weights
        """
        return np.sum(np.column_stack(results), axis=1)

    def normalize(self, combined_result, num_replicates, **kwargs): #TODO: Explain to Drew
        """
        Normalize weights

        Parameters
        ----------
        combined_result : numpy.ndarray
            Combined weights
        num_replicates : int
            Number of replicates

        Returns
        -------
        numpy.ndarray
            Normalized weights
        """
        # Replace NaN and infinite values
        combined_result = np.nan_to_num(combined_result, nan=1.0, posinf=1e10, neginf=0.0)

        # Normalize
        return combined_result / np.sum(combined_result) * len(combined_result)


class MeanEvaluator(Evaluator):
    """
    Base class for evaluators that compute means
    """

    def combine(self, *results):
        """
        Sum evaluation results

        Parameters
        ----------
        *results : numeric
            Evaluation results

        Returns
        -------
        numeric
            Sum of results
        """
        return sum(results)

    def normalize(self, combined_result, num_replicates, **kwargs):
        """
        Normalize by dividing by number of replicates

        Parameters
        ----------
        combined_result : numeric
            Combined result
        num_replicates : int
            Number of replicates

        Returns
        -------
        numeric
            Normalized result
        """
        return combined_result / num_replicates


class MSEEvaluator(MeanEvaluator):
    """
    Evaluator that computes Mean Squared Error
    """

    def evaluate(self, model, data):
        """
        Compute MSE for a model

        Parameters
        ----------
        model : function
            Weight function
        data : dict
            Data dictionary

        Returns
        -------
        float
            Mean Squared Error
        """
        # Compute weights for observed data
        weights = model(data['observed']['A'], data['observed']['X'])
        prob = weights / (1 + weights)
        # For observed data, we want prob(C=0) to be high, so MSE is (1-prob)^2
        observed_mse = np.mean((1 - prob) ** 2)

        # Compute weights for permuted data
        weights = model(data['permuted']['A'], data['permuted']['X'])
        prob = weights / (1 + weights)
        # For permuted data, we want prob(C=1) to be high, so MSE is prob^2
        permuted_mse = np.mean(prob ** 2)

        return np.mean([observed_mse, permuted_mse])


class LogLossEvaluator(MeanEvaluator):
    """
    Evaluator that computes Log Loss
    """

    EPS = 1e-15

    def evaluate(self, model, data):
        """
        Compute Log Loss for a model

        Parameters
        ----------
        model : function
            Weight function
        data : dict
            Data dictionary

        Returns
        -------
        float
            Log Loss
        """
        # Compute weights for observed data
        weights = model(data['observed']['A'], data['observed']['X'])
        prob = weights / (1 + weights)
        prob_clipped = np.clip(prob, self.EPS, 1 - self.EPS)
        # For observed data (C=0), log loss is -log(1-prob)
        observed_logloss = np.mean(-np.log(1 - prob_clipped))

        # Compute weights for permuted data
        weights = model(data['permuted']['A'], data['permuted']['X'])
        prob = weights / (1 + weights)
        prob_clipped = np.clip(prob, self.EPS, 1 - self.EPS)
        # For permuted data (C=1), log loss is -log(prob)
        permuted_logloss = np.mean(-np.log(prob_clipped))

        return np.mean([observed_logloss, permuted_logloss])


def evaluator_factory(eval_name):
    """
    Factory for creating evaluators

    Parameters
    ----------
    eval_name : str
        Name of evaluator ('mse' or 'logloss')

    Returns
    -------
    Evaluator
        Evaluator instance
    """
    eval_name = eval_name.lower()

    if eval_name == "mse":
        return MSEEvaluator()
    elif eval_name == "logloss":
        return LogLossEvaluator()
    else:
        raise ValueError(f"Unknown evaluator: {eval_name}")