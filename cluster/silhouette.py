import numpy as np
from scipy.spatial.distance import cdist


class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """

        # Check that inputs are of the correct type, shape, and contents
        if not isinstance(X, np.ndarray):
            raise TypeError('X must be a numpy array')
        if X.size == 0:
            raise ValueError('X must be non-empty')
        if X.ndim != 2:
            raise ValueError('X must be 2-dimensional')
        if X.shape[0] < 2:
            raise ValueError('X must contain at least 2 rows/observations')
        if not isinstance(y, np.ndarray):
            raise TypeError('y must be a numpy array')
        if y.size == 0:
            raise ValueError('y must be non-empty')
        if y.ndim != 1:
            raise ValueError('y must be 1-dimensional')
        if X.shape[0] != y.shape[0]:
            raise ValueError('X and y must have the same number of rows/observations')

        # Determine the unique cluster labels
        labels = np.unique(y)
        if labels.size < 2:
            raise ValueError('y must contain at least 2 clusters')

        # Initialize a and b
        a = np.zeros(X.shape[0])
        b = np.ones(X.shape[0]) * np.inf

        # Compute a and b
        for label_i in labels:
            
            # Get all points with label_i
            X_cluster_i = X[y == label_i]
            
            # Compute a (the mean intra-cluster distance for each point)
            if X_cluster_i.shape[0] > 1:

                # Compute intra-cluster pair-wise distances
                dists_ii = cdist(X_cluster_i, X_cluster_i, metric='euclidean')
                
                # Compute the mean pair-wise distance (not including self-self distance)
                mean_dists_ii = dists_ii.sum(axis=1) / (X_cluster_i.shape[0] - 1)
                
                # Update a for all points with label_i
                a[y == label_i] = mean_dists_ii
            
            # Compute b (the mean nearest-cluster distance for each point)
            for label_j in labels:
                
                # If label_j and label_i do not match, compute b
                if label_j != label_i:
                
                    # Get all points with label_j and compute pair-wise distances between points with
                    # label_i and points with label_j
                    X_cluster_j = X[y == label_j]
                    dists_ij = cdist(X_cluster_i, X_cluster_j, metric='euclidean')

                    # Compute the mean ij pair-wise distance
                    mean_dists_ij = dists_ij.mean(axis=1)

                    # Update b such that it is the minimum mean distance to any other cluster
                    b[y == label_i] = np.minimum(b[y == label_i], mean_dists_ij)

        # Compute the silhouette scores
        silhouette = (b - a) / np.maximum(a, b)
        return silhouette