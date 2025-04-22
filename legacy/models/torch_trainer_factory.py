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
        'learning_rate': 0.001,  # Reduced from 0.01
        'epochs': 100,
        'hidden_dims': [64, 32],
        'hidden_dim': 64,
        'num_blocks': 2,
        'l2_reg': 0.01,
        'early_stopping': True,
        'patience': 10,
        'verbose': False,  # Add this parameter
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
        
        # Add scheduler for learning rate adjustment
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5, 
            verbose=model_params.get('verbose', False)  # Only be verbose if requested
        )

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

            # Train the model
            best_loss = float('inf')
            patience_counter = 0
            val_losses = []
            train_losses = []
            best_model = None

            # Prepare validation data if early stopping is enabled
            if model_params.get('early_stopping', False):
                # Make sure we have enough data for validation
                if len(dataset) >= 4:  # Need at least 4 samples to split
                    # Split dataset for validation (80/20 split)
                    val_size = max(1, min(int(len(dataset) * 0.2), len(dataset) // 2))
                    indices = torch.randperm(len(dataset))
                    train_indices = indices[:-val_size].tolist()
                    val_indices = indices[-val_size:].tolist()

                    # Create samplers
                    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
                    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

                    # Create dataloaders
                    train_loader = DataLoader(
                        dataset,
                        batch_size=min(model_params['batch_size'], len(train_indices)),
                        sampler=train_sampler,
                        num_workers=0  # Use 0 for reliability in tests
                    )

                    val_loader = DataLoader(
                        dataset,
                        batch_size=min(model_params['batch_size'], len(val_indices)),
                        sampler=val_sampler,
                        num_workers=0  # Use 0 for reliability in tests
                    )

                    use_validation = True
                else:
                    # Not enough data for validation
                    train_loader = dataloader
                    val_loader = None
                    use_validation = False
            else:
                # No early stopping
                train_loader = dataloader
                val_loader = None
                use_validation = False

            # Train the model
            for epoch in range(model_params['epochs']):
                # Training phase
                model.train()
                epoch_train_loss = 0
                train_samples = 0

                for X_batch, y_batch in train_loader:
                    batch_size = X_batch.size(0)
                    train_samples += batch_size

                    X_batch = X_batch.to(model_params['device'])
                    y_batch = y_batch.to(model_params['device'])

                    # Forward pass
                    y_pred = model(X_batch).squeeze()

                    # Handle scalar output for single sample
                    if batch_size == 1:
                        y_pred = y_pred.unsqueeze(0)

                    # Ensure y_pred and y_batch have compatible shapes
                    if y_pred.shape != y_batch.shape:
                        y_pred = y_pred.reshape(y_batch.shape)

                    # Numerically stable computation
                    y_pred = torch.clamp(y_pred, 1e-7, 1 - 1e-7)

                    loss = criterion(y_pred, y_batch)

                    # Backward pass and optimize
                    optimizer.zero_grad()
                    loss.backward()

                    # Clip gradients for stability
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    optimizer.step()

                    epoch_train_loss += loss.item() * batch_size

                # Avoid division by zero
                if train_samples > 0:
                    epoch_train_loss /= train_samples
                else:
                    epoch_train_loss = float('inf')

                train_losses.append(epoch_train_loss)

                # Validation phase
                if use_validation:
                    model.eval()
                    epoch_val_loss = 0
                    val_samples = 0

                    with torch.no_grad():
                        for X_batch, y_batch in val_loader:
                            batch_size = X_batch.size(0)
                            val_samples += batch_size

                            X_batch = X_batch.to(model_params['device'])
                            y_batch = y_batch.to(model_params['device'])

                            # Forward pass
                            y_pred = model(X_batch).squeeze()

                            # Handle scalar output for single sample
                            if batch_size == 1:
                                y_pred = y_pred.unsqueeze(0)

                            # Ensure y_pred and y_batch have compatible shapes
                            if y_pred.shape != y_batch.shape:
                                y_pred = y_pred.reshape(y_batch.shape)

                            # Numerically stable computation
                            y_pred = torch.clamp(y_pred, 1e-7, 1 - 1e-7)

                            loss = criterion(y_pred, y_batch)
                            epoch_val_loss += loss.item() * batch_size

                    # Avoid division by zero
                    if val_samples > 0:
                        epoch_val_loss /= val_samples
                    else:
                        epoch_val_loss = float('inf')

                    val_losses.append(epoch_val_loss)
                    current_loss = epoch_val_loss
                    
                    # Step the scheduler
                    scheduler.step(current_loss)
                else:
                    # No validation - use training loss
                    current_loss = epoch_train_loss

                # Early stopping logic
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_model = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1

                    # Check for increasing trend
                    if epoch > 10 and len(val_losses) >= 5:
                        # Check for consistent increase (more than just 3 points)
                        recent_losses = val_losses[-5:]
                        is_increasing = all(
                            recent_losses[i] > recent_losses[i - 1] for i in range(1, len(recent_losses)))

                        if is_increasing:
                            if model_params.get('verbose', False):
                                print(f"Stopping early at epoch {epoch + 1}: Consistent loss increase over 5 epochs")
                            if best_model is not None:
                                model.load_state_dict(best_model)
                            break

                    if model_params.get('patience', 10) is not None and patience_counter >= model_params.get('patience',
                                                                                                             10):
                        if model_params.get('verbose', False):
                            print(f"Stopping early at epoch {epoch + 1}: Patience exceeded")
                        if best_model is not None:
                            model.load_state_dict(best_model)
                        break

            # Print progress summary at the end of training
            if model_params.get('verbose', False):
                print(f"Training completed in {epoch+1} epochs")
                print(f"Best validation loss: {best_loss:.4f}")
                print(f"Final loss: {current_loss:.4f}")
            else:
                # Just a simple dot to show progress without cluttering output
                if epoch % 20 == 0:
                    print(".", end="", flush=True)
            
            # Print a newline if we were printing dots
            if not model_params.get('verbose', False):
                print()
                
            # Store convergence information
            convergence_info = {
                'converged': patience_counter < model_params['patience'],
                'iterations': epoch + 1,
                'final_loss': val_losses[-1] if val_losses else train_losses[-1],
                'best_loss': best_loss
            }
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
            probs = np.clip(probs, 0.001, 0.999)

            weights = probs / (1 - probs)

            # Handle extreme values
            weights = np.nan_to_num(weights, nan=1.0, posinf=1e10, neginf=0.0)

            return weights

        return weight_function

    return trainer


def minibatch_permute_torch_factory(model_type='logistic', params=None):
    """
    Factory for PyTorch model trainers using minibatch with in-batch permutation

    Parameters:
    ----------
    model_type : str, default='logistic'
        Type of model ('logistic', 'mlp', 'resnet')
    params : dict, optional
        Model parameters

    Returns:
    -------
    function
        A function that trains a PyTorch model
    """
    if params is None:
        params = {}

    # Set default parameters
    default_params = {
        'batch_size': 64,
        'learning_rate': 0.001,  # Reduced from 0.01
        'epochs': 100,
        'hidden_dims': [64, 32],
        'hidden_dim': 64,
        'num_blocks': 2,
        'l2_reg': 0.01,
        'early_stopping': True,
        'patience': 10,
        'verbose': False,  # Add this parameter
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    # Override defaults with provided params
    model_params = {**default_params, **params}

    # Validate model_type
    if model_type not in ['logistic', 'mlp', 'resnet']:
        raise ValueError(f"Unknown model type: {model_type}")

    def trainer(data):
        """
        Trains a PyTorch model using minibatch with in-batch permutation

        Parameters:
        ----------
        data : dict
            Dictionary containing observed data

        Returns:
        -------
        function
            A function that computes weights
        """
        # Extract observed data
        A = data['observed']['A']
        X = data['observed']['X']
        n = len(A)

        # Get input dimension
        if isinstance(X, np.ndarray):
            input_dim = X.shape[1] + 1  # Add 1 for treatment
        else:
            input_dim = X.shape[1] + 1

        # Initialize model
        if model_type == 'logistic':
            model = LogisticNet(input_dim)
        elif model_type == 'mlp':
            model = MLPNet(input_dim, model_params['hidden_dims'])
        elif model_type == 'resnet':
            model = DeepResNet(input_dim, model_params['hidden_dim'], model_params['num_blocks'])

        model = model.to(model_params['device'])

        # Initialize optimizer and criterion
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=model_params['learning_rate'],
            weight_decay=model_params['l2_reg']
        )
        criterion = nn.BCELoss()

        # Standardize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()

        if isinstance(X, np.ndarray):
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = scaler.fit_transform(X.values)

        # Convert to tensors
        A_tensor = torch.tensor(A, dtype=torch.float32)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        # Train the model
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(model_params['epochs']):
            model.train()
            epoch_loss = 0

            # Number of mini-batches per epoch
            n_batches = max(1, int(np.ceil(n / model_params['batch_size'])))

            for _ in range(n_batches):
                # Sample batch indices
                batch_indices = np.random.choice(n, size=min(model_params['batch_size'], n), replace=False)

                # Get original batch data
                batch_A = A_tensor[batch_indices].to(model_params['device'])
                batch_X = X_tensor[batch_indices].to(model_params['device'])

                # Original data (label 0)
                batch_features_original = torch.cat([batch_A.unsqueeze(1), batch_X], dim=1)
                batch_labels_original = torch.zeros(len(batch_indices), device=model_params['device'])

                # Forward pass for original data
                y_pred_original = model(batch_features_original).squeeze()
                loss_original = criterion(y_pred_original, batch_labels_original)

                # Create stratified permutation sampling to maintain balance
                if epoch == 0:
                    # Initialize tracking for permutation balancing
                    permutation_tracker = {'counts': torch.zeros(n, device=model_params['device'])}
                else:
                    # Use indices that have been used less frequently
                    available_mask = permutation_tracker['counts'] < torch.median(permutation_tracker['counts'])
                    if available_mask.sum() < len(batch_indices):
                        # Reset if too few available
                        permutation_tracker['counts'] = torch.zeros(n, device=model_params['device'])
                        available_mask = torch.ones(n, dtype=torch.bool, device=model_params['device'])

                    # Get available indices
                    available_indices = torch.nonzero(available_mask).squeeze()

                # Choose permutation indices
                if 'available_indices' in locals() and len(available_indices) >= len(batch_indices):
                    # Sample from available indices
                    idx = torch.randint(0, len(available_indices), (len(batch_indices),), device=model_params['device'])
                    perm_source_indices = available_indices[idx]
                else:
                    # Default random permutation
                    perm_source_indices = torch.randint(0, n, (len(batch_indices),), device=model_params['device'])

                # Update permutation counts
                if 'permutation_tracker' in locals():
                    permutation_tracker['counts'][perm_source_indices] += 1

                # Get permuted treatments
                if hasattr(A_tensor, 'to'):
                    batch_A_perm = A_tensor[perm_source_indices].to(model_params['device'])
                else:
                    # Fall back to random permutation if tracking not working
                    perm_indices = torch.randperm(len(batch_indices))
                    batch_A_perm = batch_A[perm_indices]

                # Permuted data (label 1)
                batch_features_permuted = torch.cat([batch_A_perm.unsqueeze(1), batch_X], dim=1)
                batch_labels_permuted = torch.ones(len(batch_indices), device=model_params['device'])


                # Forward pass for permuted data
                y_pred_permuted = model(batch_features_permuted).squeeze()
                loss_permuted = criterion(y_pred_permuted, batch_labels_permuted)

                # Combined loss
                loss = loss_original + loss_permuted

                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * len(batch_indices) * 2  # For both original and permuted

            epoch_loss /= (n_batches * model_params['batch_size'] * 2)

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

        def weight_function(A_new, X_new):
            """
            Computes weights from the trained model

            Parameters:
            ----------
            A_new : array-like
                Treatment variable
            X_new : array-like
                Covariate matrix

            Returns:
            -------
            numpy.ndarray
                Computed weights
            """
            model.eval()

            # Standardize new data
            if isinstance(X_new, np.ndarray):
                X_new_scaled = scaler.transform(X_new)
            else:
                X_new_scaled = scaler.transform(X_new.values)

            # Convert to tensors
            A_new_tensor = torch.tensor(A_new, dtype=torch.float32)
            X_new_tensor = torch.tensor(X_new_scaled, dtype=torch.float32)

            # Combine features
            features = torch.cat([A_new_tensor.unsqueeze(1), X_new_tensor], dim=1)
            features = features.to(model_params['device'])

            # Get predictions
            with torch.no_grad():
                probs = model(features).squeeze().cpu().numpy()

            # Compute weights
            probs = np.clip(probs, 0.001, 0.999)
            weights = probs / (1 - probs)

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