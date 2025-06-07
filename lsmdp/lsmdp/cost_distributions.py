"""
Cost distribution functions for Linearly Solvable Markov Decision Processes (LS-MDPs).

This module provides functions to generate cost functions for LS-MDPs
using different probability distributions.
"""

import numpy as np


def generate_uniform_costs(n, low=0.0, high=1.0, seed=None):
    """
    Generate costs uniformly distributed in the range [low, high].
    
    Parameters
    ----------
    n : int
        Number of states.
    low : float, optional
        Lower bound of the uniform distribution, by default 0.0.
    high : float, optional
        Upper bound of the uniform distribution, by default 1.0.
    seed : int, optional
        Random seed for reproducibility, by default None.
        
    Returns
    -------
    numpy.ndarray
        Array of costs of shape (n,).
    """
    if seed is not None:
        np.random.seed(seed)
    
    return np.random.uniform(low, high, size=n)


def generate_gaussian_costs(n, mean=1.0, std=0.5, clip_min=0.0, seed=None):
    """
    Generate costs from a Gaussian (normal) distribution.
    
    Parameters
    ----------
    n : int
        Number of states.
    mean : float, optional
        Mean of the Gaussian distribution, by default 1.0.
    std : float, optional
        Standard deviation of the Gaussian distribution, by default 0.5.
    clip_min : float, optional
        Minimum value for costs (to ensure non-negativity), by default 0.0.
    seed : int, optional
        Random seed for reproducibility, by default None.
        
    Returns
    -------
    numpy.ndarray
        Array of costs of shape (n,).
    """
    if seed is not None:
        np.random.seed(seed)
    
    costs = np.random.normal(loc=mean, scale=std, size=n)
    if clip_min is not None:
        costs = np.clip(costs, clip_min, None)
    
    return costs


def generate_exponential_costs(n, scale=1.0, seed=None):
    """
    Generate costs from an exponential distribution.
    
    Parameters
    ----------
    n : int
        Number of states.
    scale : float, optional
        Scale parameter (inverse of rate parameter) of the exponential distribution, by default 1.0.
    seed : int, optional
        Random seed for reproducibility, by default None.
        
    Returns
    -------
    numpy.ndarray
        Array of costs of shape (n,).
    """
    if seed is not None:
        np.random.seed(seed)
    
    return np.random.exponential(scale=scale, size=n)


def generate_costs(n, distribution="uniform", **params):
    """
    Generate costs using the specified distribution.
    
    Parameters
    ----------
    n : int
        Number of states.
    distribution : str, optional
        Distribution type, one of "uniform", "gaussian", or "exponential", by default "uniform".
    **params : dict
        Additional parameters for the specific distribution.
        
    Returns
    -------
    numpy.ndarray
        Array of costs of shape (n,).
        
    Raises
    ------
    ValueError
        If an unknown distribution type is specified.
    """
    if distribution.lower() == "uniform":
        return generate_uniform_costs(n, **params)
    elif distribution.lower() in ["gaussian", "normal"]:
        return generate_gaussian_costs(n, **params)
    elif distribution.lower() == "exponential":
        return generate_exponential_costs(n, **params)
    else:
        raise ValueError(f"Unknown distribution type: {distribution}")


def set_goal_states(costs, goal_states):
    """
    Set the cost of goal states to zero.
    
    Parameters
    ----------
    costs : numpy.ndarray
        Array of costs of shape (n,).
    goal_states : list or numpy.ndarray
        Indices of goal states.
        
    Returns
    -------
    numpy.ndarray
        Modified array of costs with goal states set to zero.
    """
    costs = costs.copy()  # Create a copy to avoid modifying the original array
    for state in goal_states:
        costs[state] = 0.0
    
    return costs
