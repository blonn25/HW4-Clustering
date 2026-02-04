# write your silhouette score unit tests here

import numpy as np
from cluster import KMeans, Silhouette, make_clusters
from sklearn.metrics import silhouette_score
import pytest


def check_silhouette_score(clusters: np.ndarray, pred_labels: np.ndarray, allowed_error: float = 10e-4) -> bool:
    """
    Helper function to check the correctness of silhouette score implementation

    inputs:
        clusters: np.ndarray
            A 2D matrix where the rows are observations and columns are features.
        pred_labels: np.ndarray
            a 1D array representing the cluster labels for each of the observations in `clusters`
        allowed_error: float
            allowed difference between proposed silhouette scores and sklearn silhouette scores
    """
    my_score = Silhouette().score(clusters, pred_labels).mean()
    sk_score = silhouette_score(clusters, pred_labels)
    return abs(sk_score - my_score.item()) < allowed_error


def test_silhouette_tight():
    """
    Unit test for tightly clustered data
    """
    clusters, labels = make_clusters(scale=0.3)
    km = KMeans(k=3)
    km.fit(clusters)
    pred_labels = km.predict(clusters)
    assert check_silhouette_score(clusters, pred_labels)


def test_silhouette_loose():
    """
    Unit test for loosely clustered data
    """
    clusters, labels = make_clusters(scale=2)
    km = KMeans(k=3)
    km.fit(clusters)
    pred_labels = km.predict(clusters)
    assert check_silhouette_score(clusters, pred_labels)


def test_silhouette_many():
    """
    Unit test for data with many clusters
    """
    clusters, labels = make_clusters(k=50)
    km = KMeans(k=50)
    km.fit(clusters)
    pred_labels = km.predict(clusters)
    assert check_silhouette_score(clusters, pred_labels)