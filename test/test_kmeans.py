# Write your k-means unit tests here
#
# Note: An LLM was used to assist in writing unit tests, particularly for error handling,
#       in order to help quickly implement many tests for different error cases. However,
#       unit tests were reviewed and edited by hand to ensure they were appropriate and correct

import numpy as np
from cluster import KMeans, make_clusters
import pytest


##### A ton of initialization error handling tests

def test_kmeans_k_zero():
    """
    Unit test for invalid k parameter (k=0)
    """
    with pytest.raises(ValueError, match='k must be greater than 0'):
        KMeans(k=0)

def test_kmeans_k_negative():
    """
    Unit test for invalid k parameter (k<0)
    """
    with pytest.raises(ValueError, match='k must be greater than 0'):
        KMeans(k=-5)

def test_kmeans_k_non_integer():
    """
    Unit test for invalid k type (non-integer)
    """
    with pytest.raises(TypeError, match='k must be an integer'):
        KMeans(k=3.5)

def test_kmeans_tol_negative():
    """
    Unit test for invalid tolerance parameter (tol<0)
    """
    with pytest.raises(ValueError, match='tol must be greater than or equal to 0'):
        KMeans(k=3, tol=-0.1)

def test_kmeans_tol_non_number():
    """
    Unit test for invalid tolerance type (non-numeric)
    """
    with pytest.raises(TypeError, match='tol must be a number'):
        KMeans(k=3, tol='0.01')

def test_kmeans_max_iter_zero():
    """
    Unit test for invalid max_iter parameter (max_iter=0)
    """
    with pytest.raises(ValueError, match='max_iter must be greater than 0'):
        KMeans(k=3, max_iter=0)

def test_kmeans_max_iter_negative():
    """
    Unit test for invalid max_iter parameter (max_iter<0)
    """
    with pytest.raises(ValueError, match='max_iter must be greater than 0'):
        KMeans(k=3, max_iter=-10)

def test_kmeans_max_iter_non_integer():
    """
    Unit test for invalid max_iter type (non-integer)
    """
    with pytest.raises(TypeError, match='max_iter must be an integer'):
        KMeans(k=3, max_iter=100.5)

def test_kmeans_init_invalid():
    """
    Unit test for invalid init parameter
    """
    with pytest.raises(ValueError, match="init must be either 'random' or 'k-means\\+\\+'"):
        KMeans(k=3, init='invalid')

def test_kmeans_init_non_string():
    """
    Unit test for invalid init type (non-string)
    """
    with pytest.raises(TypeError, match="init must be a string and either 'random' or 'k-means\\+\\+'"):
        KMeans(k=3, init=123)


##### More error handling tests, but for fit()

def test_kmeans_fit_non_array():
    """
    Unit test for fit() with non-array input
    """
    km = KMeans(k=3)
    with pytest.raises(TypeError, match='Input matrix must be a numpy array'):
        km.fit([1, 2, 3, 4])

def test_kmeans_fit_empty():
    """
    Unit test for fit() with empty array
    """
    km = KMeans(k=3)
    empty_mat = np.array([])
    with pytest.raises(ValueError, match='Input matrix must be non-empty'):
        km.fit(empty_mat)

def test_kmeans_fit_1d():
    """
    Unit test for fit() with 1D input
    """
    km = KMeans(k=3)
    data_1d = np.array([1, 2, 3, 4, 5])
    with pytest.raises(ValueError, match='Input matrix must be 2-dimensional'):
        km.fit(data_1d)

def test_kmeans_fit_fewer_samples_than_k():
    """
    Unit test for fit() when number of samples < k
    """
    km = KMeans(k=10)
    data = np.random.rand(5, 2)  # 5 samples, k=10
    with pytest.raises(ValueError, match='Input matrix must have at least k rows'):
        km.fit(data)


##### Error handling tests for predict()

def test_kmeans_predict_before_fit():
    """
    Unit test for predict() called before fit()
    """
    km = KMeans(k=3)
    data = np.random.rand(10, 2)
    with pytest.raises(RuntimeError, match='KMeans model must be fit before calling predict'):
        km.predict(data)

def test_kmeans_predict_non_array():
    """
    Unit test for predict() with non-array input
    """
    km = KMeans(k=3)
    clusters, _ = make_clusters(k=3)
    km.fit(clusters)
    with pytest.raises(TypeError, match='Input matrix must be a numpy array'):
        km.predict([1, 2, 3])

def test_kmeans_predict_empty():
    """
    Unit test for predict() with empty array
    """
    km = KMeans(k=3)
    clusters, _ = make_clusters(k=3)
    km.fit(clusters)
    empty_mat = np.array([])
    with pytest.raises(ValueError, match='Input matrix must be non-empty'):
        km.predict(empty_mat)

def test_kmeans_predict_1d():
    """
    Unit test for predict() with 1D input
    """
    km = KMeans(k=3)
    clusters, _ = make_clusters(k=3)
    km.fit(clusters)
    data_1d = np.array([1, 2, 3, 4, 5])
    with pytest.raises(ValueError, match='Input matrix must be 2-dimensional'):
        km.predict(data_1d)

def test_kmeans_predict_dimension_mismatch():
    """
    Unit test for predict() with feature dimension mismatch
    """
    km = KMeans(k=3)
    clusters, _ = make_clusters(k=3, m=2)  # 2D data
    km.fit(clusters)
    
    # Try to predict on data with different number of features
    mismatched_data = np.random.rand(10, 5)  # 5 features instead of 2
    with pytest.raises(ValueError, match='Input matrix must have the same number of columns'):
        km.predict(mismatched_data)


##### Finally testing the behavior of the algorithm

def test_kmeans_tight():
    """
    Unit test for KMeans convergence on tightly clustered data
    """
    clusters, _ = make_clusters(scale=0.3, k=3)
    km = KMeans(k=3, max_iter=100)
    km.fit(clusters)
    pred = km.predict(clusters)
    
    # Check that predictions are reasonable and all clusters were assigned
    assert len(np.unique(pred)) == 3, 'Not all clusters were assigned'
    assert km.get_centroids() is not None, 'Centroids should be set after fit'
    assert km.get_centroids().shape == (3, clusters.shape[1]), 'Centroids shape mismatch'

def test_kmeans_loose():
    """
    Unit test for KMeans convergence on loosely clustered data
    """
    clusters, _ = make_clusters(scale=2, k=3)
    km = KMeans(k=3, max_iter=100)
    km.fit(clusters)
    pred = km.predict(clusters)
    
    # Check that predictions are reasonable and all clusters were assigned
    assert len(np.unique(pred)) == 3, 'Not all clusters were assigned'
    assert km.get_centroids() is not None, 'Centroids should be set after fit'
    assert km.get_centroids().shape == (3, clusters.shape[1]), 'Centroids shape mismatch'

def test_kmeans_many_k():
    """
    Unit test for KMeans with high k value
    """
    clusters, _ = make_clusters(n=500, k=20)
    km = KMeans(k=20)
    km.fit(clusters)
    pred = km.predict(clusters)
    
    # Check that predictions are reasonable and all clusters were assigned
    assert len(np.unique(pred)) == 20, 'Not all clusters were assigned'
    assert km.get_centroids() is not None, 'Centroids should be set after fit'
    assert km.get_centroids().shape == (20, clusters.shape[1]), 'Centroids shape mismatch'

def test_kmeans_single_feature():
    """
    Unit test for KMeans on single-feature data (1D features)
    """
    clusters, _ = make_clusters(n=100, m=1, k=3)
    km = KMeans(k=2)
    km.fit(clusters)
    pred = km.predict(clusters)
    
    # Check that predictions are reasonable and all clusters were assigned
    assert len(np.unique(pred)) == 2, 'Not all clusters were assigned'
    assert km.get_centroids() is not None, 'Centroids should be set after fit'
    assert km.get_centroids().shape == (2, clusters.shape[1]), 'Centroids shape mismatch'

def test_kmeans_high_dim():
    """
    Unit test for KMeans on high-dimensional data
    """
    clusters, _ = make_clusters(n=100, m=100, k=3)
    km = KMeans(k=3)
    km.fit(clusters)
    pred = km.predict(clusters)
    
    # Check that predictions are reasonable and all clusters were assigned
    assert len(np.unique(pred)) == 3, 'Not all clusters were assigned'
    assert km.get_centroids() is not None, 'Centroids should be set after fit'
    assert km.get_centroids().shape == (3, clusters.shape[1]), 'Centroids shape mismatch'