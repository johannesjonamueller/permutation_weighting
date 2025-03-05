# Permutation Weighting

A Python implementation of Permutation Weighting (PW) for causal inference, based on the paper by Arbour, Dimmery, and Sondhi (2020).

## Overview

Permutation Weighting is a novel method for estimating balancing weights in observational causal inference that transforms the problem of finding balanced weights into a binary classification task.

### Key Features

- **Flexible Treatment Types**: Works with binary, multi-valued, and continuous treatments
- **Machine Learning Integration**: Uses standard classification algorithms for weight estimation
- **Cross-Validation Support**: Enables hyperparameter tuning through standard machine learning techniques
- **Theoretical Guarantees**: Provides bounds on bias and variance of causal estimates

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


The package supports advanced neural network architectures for permutation weighting using PyTorch.

### Installation with PyTorch Support

```bash
pip install permutation-weighting[torch]
```
### Installation with PyTorch Support

We addd a installation option in development mode 'install_dev.py'. 
```bash
python install_dev.py
```


## Quick Start

```python
import numpy as np
from permutation_weighting import PW
import statsmodels.api as sm

# Generate example data
n = 1000
X = np.random.normal(size=(n, 4))
propensity = 1 / (1 + np.exp(X[:, 0] - 0.5 * X[:, 1] + 0.25 * X[:, 2] + 0.1 * X[:, 3]))
A = np.random.binomial(1, propensity, size=n)
Y = 210 + 27.4 * X[:, 0] + 13.7 * X[:, 1] + 13.7 * X[:, 2] + 13.7 * X[:, 3] + np.random.normal(size=n)

# Fit permutation weighting model
model = PW(A, X, num_replicates=100)

# Estimate average treatment effect
result = sm.WLS(Y, sm.add_constant(A), weights=model['weights']).fit()
print(result.summary())
```

## Supported Methods

### Classifiers
- Logistic Regression
- Gradient Boosting
- Stochastic Gradient Descent (SGD)
- Neural Network

### Estimands
- Average Treatment Effect (ATE)
- Average Treatment on the Treated (ATT)

## Key Parameters

- `A`: Treatment variable
- `X`: Covariate matrix
- `classifier`: Classification method ('logit', 'boosting', 'sgd_logit', 'neural_net')
- `estimand`: Target causal estimand ('ATE' or 'ATT')
- `num_replicates`: Number of permutations to generate weights

## Performance Characteristics

- Reduces bias under model misspecification
- Provides principled hyperparameter tuning
- Works across different treatment types
- Computationally efficient

## Citation

If you use this package, please cite the original paper:

Arbour, D., Dimmery, D., & Sondhi, A. (2020). Permutation Weighting. arXiv preprint arXiv:1901.01230.

## Package Structure

```
permutation_weighting/
├── permutation_weighting/
│   ├── __init__.py          # Package initialization
│   ├── estimator.py         # Main estimation algorithm (PW function)
│   ├── evaluator.py         # Evaluation metrics and utilities
│   ├── utils.py             # Helper functions
│   │
│   ├── data/                # Data handling module
│   │   ├── __init__.py
│   │   ├── data_factory.py  # Data generation and preparation
│   │   └── data_validation.py  # Input validation
│   │
│   └── models/              # Model training module
│       ├── __init__.py
│       ├── trainer_factory.py    # Standard model training logic
│       ├── torch_trainer_factory.py   # Torch model training 
│       └── sgd_trainer_factory.py # SGD-based model training
│
├── tests/                   # Unit tests
│   ├── __init__.py
│   ├── test_data_factory.py
│   ├── test_data_validation.py
│   ├── test_estimator.py
│   ├── test_evaluator.py
│   ├── test_trainer_factory.py
│   ├── test_torch_trainer_factory.py
│   └── test_sgd_trainer_factory.py
│
├── examples/                # Example notebooks and scripts
│   ├── kang_schafer_simulation.ipynb
│   ├── comparison_with_r_package.ipynb
│   └── sgd_performance_benchmarks.ipynb
│
├── README.md                # Package documentation
├── setup.py                 # Package setup and distribution
├── pyproject.toml           # Build system requirements
└── LICENSE                  # License information
```

### Key Modules Explained

- **estimator.py**: Contains the core `PW()` function that estimates balancing weights
- **evaluator.py**: Provides evaluation metrics like Mean Squared Error and Log Loss
- **data/data_factory.py**: Generates balanced datasets for different estimands (ATE, ATT)
- **data/data_validation.py**: Validates input data
- **models/trainer_factory.py**: Creates model trainers for different classifiers
- **models/sgd_trainer_factory.py**: Implements Stochastic Gradient Descent-based training


## License

MIT License