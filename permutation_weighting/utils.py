"""
Utility functions for permutation weighting.
"""

import numpy as np
import warnings


def muffle_warnings(func, *patterns):
    """
    Execute a function while suppressing specific warnings

    Parameters
    ----------
    func : callable
        Function to execute
    *patterns : str
        Warning message patterns to suppress

    Returns
    -------
    any
        Result of the function
    """
    with warnings.catch_warnings():
        for pattern in patterns:
            warnings.filterwarnings("ignore", message=pattern)
        return func()