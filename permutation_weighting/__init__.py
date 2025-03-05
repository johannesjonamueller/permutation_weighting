"""
Permutation Weighting

A Python implementation of the permutation weighting method for causal inference.
Based on: "Permutation Weighting" by David Arbour, Drew Dimmery, and Arjun Sondhi (2020).
"""
from .estimator import  PW

__version__ = "0.1.0"
__all__ = ["PW"]

# from .models.trainer_factory import get_trainer_factory, logit_factory, boosting_factory
# from .models.sgd_trainer_factory import sgd_logit_factory, neural_net_factory, minibatch_trainer_factory
#
# # Import PyTorch trainers conditionally to avoid errors if PyTorch is not installed
# try:
#     from .models.torch_trainer_factory import (
#         torch_trainer_factory, logistic_torch_factory,
#         mlp_torch_factory, resnet_torch_factory
#     )
#     TORCH_AVAILABLE = True
# except ImportError:
#     TORCH_AVAILABLE = False
#
# __all__ = [
#     "get_trainer_factory", "logit_factory", "boosting_factory",
#     "sgd_logit_factory", "neural_net_factory", "minibatch_trainer_factory"
# ]
#
# if TORCH_AVAILABLE:
#     __all__.extend([
#         "torch_trainer_factory", "logistic_torch_factory",
#         "mlp_torch_factory", "resnet_torch_factory"
#     ])