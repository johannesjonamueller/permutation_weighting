# Permutation Weighting

A Python implementation of Permutation Weighting (PW) for causal inference, based on the paper Permutation Weighting by David Arbour, Drew Dimmery, Arjun Sondhi (2020) https://arxiv.org/abs/1901.01230

## Overview

Permutation Weighting is a novel method for estimating balancing weights in observational causal inference that transforms the problem of finding balanced weights into a binary classification task.

### Key Features

- **Flexible Treatment Types**: Works with both binary and continuous treatments
- **Machine Learning Integration**: Uses standard classification algorithms for weight estimation
- **Theoretical Guarantees**: Provides bounds on bias and variance of causal estimates
- **Multiple Classifier Options**: Supports logistic regression, gradient boosting, SGD, and neural networks

## Theoretical Background

In observational studies, causal inference requires rendering treatments independent of observed covariates - a property known as "balance". Traditional methods like Inverse Propensity Score Weighting (IPSW) often struggle with model misspecification.

Permutation Weighting addresses these challenges by:

1. Creating a balanced dataset through treatment permutation
2. Transforming the balance problem into a binary classification task
3. Estimating importance sampling weights directly

### Core Innovations

- Represents balanced datasets by permuting observed treatments
- Uses binary classification to estimate density ratios
- Provides a unified framework for different balancing weight methods
- Enables treatment effect estimation across various treatment types

## Installation

```bash
pip install permutation-weighting
```

### Development Installation

For development purposes, use the provided installation script:

```bash
python install_dev.py
```

## Quick Start

```python
import numpy as np
from permutation_weighting import PW
import statsmodels.api as sm

# Generate example data (Kang and Schafer simulation)
n = 1000
X = np.random.normal(size=(n, 4))
propensity = 1 / (1 + np.exp(X[:, 0] - 0.5 * X[:, 1] + 0.25 * X[:, 2] + 0.1 * X[:, 3]))
A = np.random.binomial(1, propensity, size=n)
Y = 210 + 27.4 * X[:, 0] + 13.7 * X[:, 1] + 13.7 * X[:, 2] + 13.7 * X[:, 3] + np.random.normal(size=n)

# Only observe masked versions of the covariates
X_mis = np.column_stack([
    np.exp(X[:, 0] / 2),
    X[:, 1] * (1 + np.exp(X[:, 0])) ** (-1) + 10,
    (X[:, 0] * X[:, 2] / 25 + 0.6) ** 3,
    (X[:, 1] + X[:, 3] + 20) ** 2
])

# Fit permutation weighting model
model = PW(A, X_mis, num_replicates=100)

# Estimate average treatment effect
result = sm.WLS(Y, sm.add_constant(A), weights=model['weights']).fit()
print(result.summary())
```

## Supported Methods

### Classifiers
- Logistic Regression (`'logit'`)
- Gradient Boosting (`'boosting'`)
- Stochastic Gradient Descent (`'sgd'`)
- Neural Network (`'mlp'`)

### Estimands
- Average Treatment Effect (ATE)
- Average Treatment on the Treated (ATT)

## Key Parameters

- `A`: Treatment variable (binary or continuous)
- `X`: Covariate matrix
- `classifier`: Classification method (`'logit'`, `'boosting'`, `'sgd'`, `'mlp'`)
- `estimand`: Target causal estimand (`'ATE'` or `'ATT'`)
- `num_replicates`: Number of permutations to generate weights
- `classifier_params`: Dictionary of parameters to pass to the classifier
- `estimand_params`: Dictionary of parameters for the estimand

## Performance Characteristics

- Reduces bias under model misspecification
- Works with both binary and continuous treatments
- Computationally efficient
- Improves over traditional weighting methods in many scenarios

## Citation

If you use this package, please cite the original paper:

Arbour, D., Dimmery, D., & Sondhi, A. (2020). Permutation Weighting. arXiv preprint arXiv:1901.01230.

## Package Structure

```
permutation_weighting/
├── __init__.py          # Package initialization
├── estimator.py         # Main estimation algorithm (PW function)
├── evaluator.py         # Evaluation metrics and utilities
├── utils.py             # Helper functions
├── data_factory.py      # Data generation and preparation
├── data_validation.py   # Input validation
└── trainer_factory.py   # Model training logic
```

### Key Modules Explained

- **estimator.py**: Contains the core `PW()` function that estimates balancing weights
- **evaluator.py**: Provides evaluation metrics like Mean Squared Error and Log Loss
- **data_factory.py**: Generates balanced datasets for different estimands (ATE, ATT)
- **data_validation.py**: Validates input data
- **trainer_factory.py**: Creates model trainers for different classifiers

## License

MIT License
