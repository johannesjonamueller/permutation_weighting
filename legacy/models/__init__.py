# permutation_weighting/models/__init__.py
"""
Model trainer factories for permutation weighting.
"""

# from trainer_factory import get_trainer_factory, logit_factory, boosting_factory
# from sgd_trainer_factory import sgd_logit_factory, neural_net_factory, minibatch_trainer_factory
#
# # Import PyTorch trainers conditionally to avoid errors if PyTorch is not installed
# try:
#     from torch_trainer_factory import (
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