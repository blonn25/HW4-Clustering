# Write your silhouette score unit tests here
#
# Note: An LLM was used to assist in writing unit tests, particularly for error handling,
#       in order to help quickly implement many tests for different error cases. However,
#       unit tests were reviewed and edited by hand to ensure they were appropriate and correct

import numpy as np
from cluster import Silhouette, make_clusters
from sklearn.metrics import silhouette_score
import pytest


def check_silhouette_score(clusters: np.ndarray, labels: np.ndarray, allowed_error: float = 10e-3) -> bool:
    """
    Helper function to check the correctness of silhouette score implementation

    inputs:
        clusters: np.ndarray
            A 2D matrix where the rows are observations and columns are features.
        labels: np.ndarray
            a 1D array representing the cluster labels for each of the observations in `clusters`
        allowed_error: float
            allowed difference between proposed silhouette scores and sklearn silhouette scores
    """
    my_score = Silhouette().score(clusters, labels).mean()
    sk_score = silhouette_score(clusters, labels)
    return abs(sk_score - my_score.item()) < allowed_error


##### Error Handling Tests

def test_silhouette_X_non_array():
    """
    Unit test for score() with non-array X input
    """
    y = np.array([0, 0, 1, 1])
    with pytest.raises(TypeError, match='X must be a numpy array'):
        Silhouette().score([1, 2, 3, 4], y)

def test_silhouette_X_empty():
    """
    Unit test for score() with empty X array
    """
    y = np.array([])
    X = np.array([])
    with pytest.raises(ValueError, match='X must be non-empty'):
        Silhouette().score(X, y)

def test_silhouette_X_1d():
    """
    Unit test for score() with 1D X input
    """
    X = np.array([1, 2, 3, 4])
    y = np.array([0, 0, 1, 1])
    with pytest.raises(ValueError, match='X must be 2-dimensional'):
        Silhouette().score(X, y)

def test_silhouette_y_non_array():
    """
    Unit test for score() with non-array y input
    """
    X = np.random.rand(4, 2)
    with pytest.raises(TypeError, match='y must be a numpy array'):
        Silhouette().score(X, [0, 0, 1, 1])

def test_silhouette_y_empty():
    """
    Unit test for score() with empty y array
    """
    X = np.random.rand(4, 2)
    y_empty = np.array([])
    with pytest.raises(ValueError, match='y must be non-empty'):
        Silhouette().score(X, y_empty)

def test_silhouette_y_2d():
    """
    Unit test for score() with 2D y input (should be 1D)
    """
    X = np.random.rand(4, 2)
    y_2d = np.array([[0, 0], [1, 1]])
    with pytest.raises(ValueError, match='y must be 1-dimensional'):
        Silhouette().score(X, y_2d)

def test_silhouette_X_y_length_mismatch():
    """
    Unit test for score() when X and y have different number of observations
    """
    X = np.random.rand(10, 2)
    y = np.array([0, 0, 1, 1])  # Only 4 labels for 10 observations
    with pytest.raises(ValueError, match='X and y must have the same number of rows'):
        Silhouette().score(X, y)

def test_silhouette_X_insufficient_rows():
    """
    Unit test for score() with only 1 observation
    """
    X = np.array([[1.0, 2.0]])  # Only 1 observation
    y = np.array([0])
    with pytest.raises(ValueError, match='X must contain at least 2 rows'):
        Silhouette().score(X, y)

def test_silhouette_single_cluster():
    """
    Unit test for score() with only 1 cluster
    """
    X = np.random.rand(10, 2)
    y = np.zeros(10, dtype=int)  # All points in cluster 0
    with pytest.raises(ValueError, match='y must contain at least 2 clusters'):
        Silhouette().score(X, y)


##### Sklearn Comparison Tests 

def test_silhouette_tight():
    """
    Unit test for score() with tightly clustered data comparing against sklearn
    """
    clusters, labels = make_clusters(scale=0.3)
    assert check_silhouette_score(clusters, labels)

def test_silhouette_loose():
    """
    Unit test for score() with loosely clustered data comparing against sklearn
    """
    clusters, labels = make_clusters(scale=2)
    assert check_silhouette_score(clusters, labels)

def test_silhouette_many_k():
    """
    Unit test for score() with many clusters comparing against sklearn
    """
    clusters, labels = make_clusters(k=100, n=5000)
    assert check_silhouette_score(clusters, labels)

def test_silhouette_single_feature():
    """
    Unit test for score() on single-feature (1D feature) data
    """
    clusters, labels = make_clusters(m=1)
    assert check_silhouette_score(clusters, labels)

def test_silhouette_high_dim():
    """
    Unit test for score() on high-dimensional data (50 features)
    """
    clusters, labels = make_clusters(n=100, m=100, k=3)
    assert check_silhouette_score(clusters, labels)