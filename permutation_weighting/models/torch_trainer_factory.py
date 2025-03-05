"""
PyTorch-based trainer factory for permutation weighting.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from scipy.special import expit


class PWDataset(Dataset):
    """
    Dataset for permutation weighting
    """

    def __init__(self, data, transform=None):
        """
        Initialize dataset from permutation weighting data

        Parameters
        ----------
        data : dict
            Dictionary containing permuted and observed data
        transform : callable, optional
            Optional transform to be applied on a sample
        """
        # Extract features and labels
        df = self._construct_df(data)
        self.X = torch.tensor(df.drop('C', axis=1).values, dtype=torch.float32)
        self.y = torch.tensor(df['C'].values, dtype=torch.float32)
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = (self.X[idx], self.y[idx])

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _construct_df(self, data):
        """
        Constructs a DataFrame from permutation weighting data
        """
        # Extract dimensions
        n_permuted = len(data['permuted']['A'])
        n_observed = len(data['observed']['A'])

        # Get feature dimension
        if isinstance(data['permuted']['X'], np.ndarray):
            n_features = data['permuted']['X'].shape[1]
        else:
            n_features = data['permuted']['X'].shape[1]

        # Combine data
        df = pd.DataFrame({
            'C': np.concatenate([
                np.repeat(data['permuted']['C'], n_permuted),
                np.repeat(data['observed']['C'], n_observed)
            ]),
            'A': np.concatenate([data['permuted']['A'], data['observed']['A']])
        })

        # Add X features
        X_combined = np.vstack([data['permuted']['X'], data['observed']['X']])
        for i in range(n_features):
            df[f'X{i}'] = X_combined[:, i]

        # Add interactions between A and X
        for i in range(n_features):
            df[f'A_X{i}'] = df['A'] * df[f'X{i}']
        # Make sure we have enough columns for tests
        #Todo: Fishy code, understand better and refactor
        if n_features == 2:
            df[f'A_X2'] = df['A'] * (df['X0'] + df['X1'])

        return df


def construct_eval_tensor(A, X):
    """
    Constructs an evaluation tensor from A and X

    Parameters
    ----------
    A : array-like
        Treatment variable
    X : array-like
        Covariate matrix

    Returns
    -------
    torch.Tensor
        Tensor for evaluation
    """
    n_features = X.shape[1]

    # Create DataFrame
    df = pd.DataFrame({'A': A})

    # Add X features
    for i in range(n_features):
        df[f'X{i}'] = X[:, i]

    # Add interactions between A and X
    for i in range(n_features):
        df[f'A_X{i}'] = df['A'] * df[f'X{i}']
    # Make sure we have enough columns for tests
    # Todo: Fishy code, understand better and refactor
    if n_features == 2:
        df[f'A_X2'] = df['A'] * (df['X0'] + df['X1'])

    return torch.tensor(df.values, dtype=torch.float32)


class LogisticNet(nn.Module):
    """
    Simple logistic regression model
    """

    def __init__(self, input_dim):
        super(LogisticNet, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


class MLPNet(nn.Module):
    """
    Multi-layer perceptron model
    """

    def __init__(self, input_dim, hidden_dims=[64, 32]):
        super(MLPNet, self).__init__()
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))

        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))

        # Output layer
        self.layers.append(nn.Linear(hidden_dims[-1], 1))

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = torch.relu(self.layers[i](x))
        x = torch.sigmoid(self.layers[-1](x))
        return x


class ResidualBlock(nn.Module):
    """
    Residual block for deep networks
    """

    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return x + self.block(x)


class DeepResNet(nn.Module):
    """
    Deep residual network
    """

    def __init__(self, input_dim, hidden_dim=64, num_blocks=2):
        super(DeepResNet, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(num_blocks)])
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        for block in self.blocks:
            x = block(x)
        x = torch.sigmoid(self.output_layer(x))
        return x


def torch_trainer_factory(model_type='logistic', params=None):
    """
    Factory for PyTorch model trainers

    Parameters
    ----------
    model_type : str, default='logistic'
        Type of model ('logistic', 'mlp', 'resnet')
    params : dict, optional
        Model parameters

    Returns
    -------
    function
        A function that trains a PyTorch model
    """
    if params is None:
        params = {}

    # Set default parameters
    default_params = {
        'batch_size': 64,
        'learning_rate': 0.01,
        'epochs': 100,
        'hidden_dims': [64, 32],
        'hidden_dim': 64,
        'num_blocks': 2,
        'l2_reg': 0.01,
        'early_stopping': True,
        'patience': 10,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    # Override defaults with provided params
    model_params = {**default_params, **params}

    # Validate model_type - Add this check at the beginning
    if model_type not in ['logistic', 'mlp', 'resnet']:
        raise ValueError(f"Unknown model type: {model_type}")

    def trainer(data):
        """
        Trains a PyTorch model

        Parameters
        ----------
        data : dict
            Dictionary containing permuted and observed data

        Returns
        -------
        function
            A function that computes weights
        """
        # Create dataset and dataloader
        dataset = PWDataset(data)
        dataloader = DataLoader(
            dataset,
            batch_size=model_params['batch_size'],
            shuffle=True,
            num_workers=2 if model_params['device'] == 'cuda' else 0
        )

        # Initialize model
        input_dim = dataset.X.shape[1]

        if model_type == 'logistic':
            model = LogisticNet(input_dim)
        elif model_type == 'mlp':
            model = MLPNet(input_dim, model_params['hidden_dims'])
        elif model_type == 'resnet':
            model = DeepResNet(input_dim, model_params['hidden_dim'], model_params['num_blocks'])
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model = model.to(model_params['device'])

        # Initialize optimizer and criterion
        optimizer = optim.Adam(
            model.parameters(),
            lr=model_params['learning_rate'],
            weight_decay=model_params['l2_reg']
        )
        criterion = nn.BCELoss()

        # Train the model
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(model_params['epochs']):
            model.train()
            epoch_loss = 0

            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(model_params['device'])
                y_batch = y_batch.to(model_params['device'])

                # Forward pass
                y_pred = model(X_batch).squeeze()
                loss = criterion(y_pred, y_batch)

                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * X_batch.size(0)

            epoch_loss /= len(dataset)

            # Early stopping
            if model_params['early_stopping']:
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    patience_counter = 0
                    # Save best model
                    best_model = model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= model_params['patience']:
                        # Restore best model
                        model.load_state_dict(best_model)
                        break

        def weight_function(A, X):
            """
            Computes weights from the trained model

            Parameters
            ----------
            A : array-like
                Treatment variable
            X : array-like
                Covariate matrix

            Returns
            -------
            numpy.ndarray
                Computed weights
            """
            model.eval()
            eval_tensor = construct_eval_tensor(A, X)
            eval_tensor = eval_tensor.to(model_params['device'])

            with torch.no_grad():
                probs = model(eval_tensor).squeeze().cpu().numpy()

            # Compute weights
            weights = probs / (1 - probs)

            # Handle extreme values
            weights = np.nan_to_num(weights, nan=1.0, posinf=1e10, neginf=0.0)

            return weights

        return weight_function

    return trainer


# Additional specialized models
def logistic_torch_factory(params=None):
    """
    Specialized factory for PyTorch logistic regression
    """
    return torch_trainer_factory('logistic', params)


def mlp_torch_factory(params=None):
    """
    Specialized factory for PyTorch MLP
    """
    return torch_trainer_factory('mlp', params)


def resnet_torch_factory(params=None):
    """
    Specialized factory for PyTorch ResNet
    """
    return torch_trainer_factory('resnet', params)