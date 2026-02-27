"""
Data generation utilities for the ML Masterclass.

Provides functions to create synthetic datasets with controllable properties
for regression, classification, and clustering tasks.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, make_moons, make_classification, make_regression


def make_regression_dataset(
    n_samples=200,
    n_features=5,
    n_informative=3,
    noise=10.0,
    n_outliers=0,
    random_state=42,
    as_dataframe=True,
):
    """Generate a regression dataset with controllable noise and outliers.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    n_features : int
        Total number of features.
    n_informative : int
        Number of informative features.
    noise : float
        Standard deviation of Gaussian noise on the target.
    n_outliers : int
        Number of outlier samples to inject (target multiplied by large factor).
    random_state : int
        Seed for reproducibility.
    as_dataframe : bool
        If True, return a pandas DataFrame; otherwise return arrays.

    Returns
    -------
    X, y or DataFrame
    """
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        random_state=random_state,
    )

    # Inject outliers
    if n_outliers > 0:
        rng = np.random.RandomState(random_state)
        outlier_idx = rng.choice(n_samples, size=min(n_outliers, n_samples), replace=False)
        y[outlier_idx] *= rng.uniform(3, 6, size=len(outlier_idx))

    if as_dataframe:
        feature_names = [f"feature_{i}" for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_names)
        df["target"] = y
        return df
    return X, y


def make_classification_dataset(
    n_samples=500,
    n_features=10,
    n_informative=5,
    n_classes=2,
    imbalance_ratio=1.0,
    random_state=42,
    as_dataframe=True,
):
    """Generate a classification dataset with controllable imbalance.

    Parameters
    ----------
    n_samples : int
        Total number of samples.
    n_features : int
        Total number of features.
    n_informative : int
        Number of informative features.
    n_classes : int
        Number of classes.
    imbalance_ratio : float
        Ratio of majority to minority class (1.0 = balanced, 5.0 = 5:1 ratio).
    random_state : int
        Seed for reproducibility.
    as_dataframe : bool
        If True, return a pandas DataFrame; otherwise return arrays.

    Returns
    -------
    X, y or DataFrame
    """
    if n_classes == 2 and imbalance_ratio != 1.0:
        # Calculate weights for imbalanced classes
        w_minority = 1.0 / (1.0 + imbalance_ratio)
        w_majority = imbalance_ratio / (1.0 + imbalance_ratio)
        weights = [w_majority, w_minority]
    else:
        weights = None

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=2,
        n_classes=n_classes,
        weights=weights,
        random_state=random_state,
        flip_y=0.03,
    )

    if as_dataframe:
        feature_names = [f"feature_{i}" for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_names)
        df["target"] = y
        return df
    return X, y


def make_clustering_blobs(
    n_samples=300,
    n_centers=3,
    cluster_std=1.0,
    random_state=42,
    as_dataframe=True,
):
    """Generate isotropic Gaussian blobs for clustering.

    Parameters
    ----------
    n_samples : int
        Total number of samples.
    n_centers : int
        Number of cluster centers.
    cluster_std : float
        Standard deviation of clusters.
    random_state : int
        Seed for reproducibility.
    as_dataframe : bool
        If True, return a pandas DataFrame.

    Returns
    -------
    X, y or DataFrame
    """
    X, y = make_blobs(
        n_samples=n_samples,
        centers=n_centers,
        cluster_std=cluster_std,
        random_state=random_state,
    )

    if as_dataframe:
        df = pd.DataFrame(X, columns=["x1", "x2"])
        df["cluster"] = y
        return df
    return X, y


def make_clustering_moons(
    n_samples=300,
    noise=0.1,
    random_state=42,
    as_dataframe=True,
):
    """Generate two interleaving half-moon shapes for clustering.

    Parameters
    ----------
    n_samples : int
        Total number of samples.
    noise : float
        Standard deviation of Gaussian noise.
    random_state : int
        Seed for reproducibility.
    as_dataframe : bool
        If True, return a pandas DataFrame.

    Returns
    -------
    X, y or DataFrame
    """
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)

    if as_dataframe:
        df = pd.DataFrame(X, columns=["x1", "x2"])
        df["cluster"] = y
        return df
    return X, y


def make_anisotropic_blobs(
    n_samples=300,
    random_state=42,
    as_dataframe=True,
):
    """Generate anisotropically distributed blobs (stretched clusters).

    Creates 3 blobs and applies a linear transformation to stretch them,
    making them harder for simple KMeans to cluster correctly.

    Parameters
    ----------
    n_samples : int
        Total number of samples.
    random_state : int
        Seed for reproducibility.
    as_dataframe : bool
        If True, return a pandas DataFrame.

    Returns
    -------
    X, y or DataFrame
    """
    X, y = make_blobs(n_samples=n_samples, centers=3, random_state=random_state)
    # Apply anisotropic transformation
    transformation = np.array([[0.6, -0.6], [-0.4, 0.8]])
    X = X @ transformation

    if as_dataframe:
        df = pd.DataFrame(X, columns=["x1", "x2"])
        df["cluster"] = y
        return df
    return X, y
