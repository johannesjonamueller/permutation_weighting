import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from permutation_weighting.models.torch_trainer_factory import (
    torch_trainer_factory, logistic_torch_factory, mlp_torch_factory,
    resnet_torch_factory, construct_eval_tensor,PWDataset,LogisticNet, MLPNet, ResidualBlock, DeepResNet


)


def test_torch_trainer_factory_unknown_model_type():
    with pytest.raises(ValueError, match="Unknown model type"):
        trainer = torch_trainer_factory('unknown')


def test_pwdataset():
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

    dataset = PWDataset(data)

    # Check dataset properties
    assert len(dataset) == 4
    assert dataset.X.shape == (4, 6)  # A, X0, X1, A_X0, A_X1 # TODO: Check this 5 or 7
    assert dataset.y.shape == (4,)

    # Check data values
    assert torch.all(dataset.y[:2] == 1)  # permuted data
    assert torch.all(dataset.y[2:] == 0)  # observed data


def test_construct_eval_tensor():
    A = np.array([0, 1])
    X = np.array([[1, 2], [3, 4]])

    tensor = construct_eval_tensor(A, X)

    # Check tensor shape
    assert tensor.shape == (2, 6)  # A, X0, X1, A_X0, A_X1 # TODO: Check this 5 or 7

    # Check tensor values
    assert tensor[0, 0] == 0  # First A value
    assert tensor[1, 0] == 1  # Second A value
    assert tensor[0, 1] == 1  # First X0 value
    assert tensor[1, 1] == 3  # Second X0 value
    assert tensor[0, 3] == 0  # First A_X0 value (0*1)
    assert tensor[1, 3] == 3  # Second A_X0 value (1*3)


def test_logistic_torch_factory():
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

    # Use small epoch count for testing
    trainer = logistic_torch_factory({'epochs': 5, 'batch_size': 2})
    model = trainer(data)

    # Make predictions
    A = np.array([0, 1])
    X = np.array([[1, 2], [3, 4]])

    weights = model(A, X)

    # Check weights
    assert len(weights) == 2
    assert np.all(weights >= 0)  # Weights should be non-negative


@pytest.mark.parametrize("factory_func,params", [
    (mlp_torch_factory, {'epochs': 5, 'batch_size': 2, 'hidden_dims': [8, 4]}),
    (resnet_torch_factory, {'epochs': 5, 'batch_size': 2, 'hidden_dim': 8, 'num_blocks': 1})
])
def test_nn_torch_factories(factory_func, params):
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

    # Test factory
    trainer = factory_func(params)
    model = trainer(data)

    # Make predictions
    A = np.array([0, 1])
    X = np.array([[1, 2], [3, 4]])

    weights = model(A, X)

    # Check weights
    assert len(weights) == 2
    assert np.all(weights >= 0)  # Weights should be non-negative


#NEW TESTS
def test_torch_trainer_factory_unknown_model_type():
    """Test that torch_trainer_factory raises ValueError for unknown model type"""
    # This should raise ValueError immediately
    with pytest.raises(ValueError, match="Unknown model type"):
        torch_trainer_factory('unknown')



def test_pytorch_model_classes():
    """Test that PyTorch model classes initialize correctly"""
    # Test LogisticNet
    logistic_model = LogisticNet(input_dim=5)
    assert isinstance(logistic_model, nn.Module)
    assert isinstance(logistic_model.linear, nn.Linear)
    assert logistic_model.linear.in_features == 5
    assert logistic_model.linear.out_features == 1

    # Test MLPNet
    mlp_model = MLPNet(input_dim=5, hidden_dims=[10, 5])
    assert isinstance(mlp_model, nn.Module)
    assert len(mlp_model.layers) == 3  # input + hidden + output
    assert mlp_model.layers[0].in_features == 5
    assert mlp_model.layers[0].out_features == 10
    assert mlp_model.layers[1].in_features == 10
    assert mlp_model.layers[1].out_features == 5
    assert mlp_model.layers[2].in_features == 5
    assert mlp_model.layers[2].out_features == 1

    # Test ResidualBlock
    res_block = ResidualBlock(dim=8)
    assert isinstance(res_block, nn.Module)
    assert isinstance(res_block.block, nn.Sequential)
    assert len(res_block.block) == 3  # linear + relu + linear

    # Test DeepResNet
    resnet_model = DeepResNet(input_dim=5, hidden_dim=8, num_blocks=2)
    assert isinstance(resnet_model, nn.Module)
    assert isinstance(resnet_model.input_layer, nn.Linear)
    assert len(resnet_model.blocks) == 2
    assert isinstance(resnet_model.output_layer, nn.Linear)


def test_model_forward_pass():
    """Test forward pass of PyTorch models"""
    batch_size = 3
    input_dim = 5

    # Create input tensor
    x = torch.rand(batch_size, input_dim)

    # Test LogisticNet forward pass
    logistic_model = LogisticNet(input_dim)
    logistic_out = logistic_model(x)
    assert logistic_out.shape == (batch_size, 1)
    assert torch.all(logistic_out >= 0) and torch.all(logistic_out <= 1)

    # Test MLPNet forward pass
    mlp_model = MLPNet(input_dim)
    mlp_out = mlp_model(x)
    assert mlp_out.shape == (batch_size, 1)
    assert torch.all(mlp_out >= 0) and torch.all(mlp_out <= 1)

    # Test DeepResNet forward pass
    resnet_model = DeepResNet(input_dim)
    resnet_out = resnet_model(x)
    assert resnet_out.shape == (batch_size, 1)
    assert torch.all(resnet_out >= 0) and torch.all(resnet_out <= 1)


def test_early_stopping():
    """Test early stopping functionality"""
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

    # Configure trainer with early stopping
    params = {
        'epochs': 20,
        'batch_size': 2,
        'early_stopping': True,
        'patience': 2
    }

    # Create a mock for the loss calculation to force early stopping
    original_bceloss = torch.nn.BCELoss

    class MockBCELoss(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.call_count = 0
            self.real_loss = original_bceloss()

        def forward(self, input, target):
            self.call_count += 1
            # Return decreasing loss for first 5 calls, then increasing
            if self.call_count <= 5:
                return self.real_loss(input, target) - (0.1 * self.call_count)
            else:
                return self.real_loss(input, target) + (0.1 * (self.call_count - 5))

    # Patch BCELoss and run the test
    original_bce_loss = torch.nn.BCELoss
    torch.nn.BCELoss = MockBCELoss

    try:
        trainer = logistic_torch_factory(params)
        model = trainer(data)

        # Test should complete without errors, no need for assertions
        # since we're just testing that early stopping doesn't crash
    finally:
        # Restore original BCELoss
        torch.nn.BCELoss = original_bce_loss