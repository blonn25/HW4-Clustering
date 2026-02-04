import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """

        # Check that k is of the correct type and within the expected range
        if not isinstance(k, int):
            raise TypeError('k must be an integer')
        if k <= 0:
            raise ValueError('k must be greater than 0')
        
        # Check that tol is of the correct type and within the expected range
        if not isinstance(tol, (int, float)):
            raise TypeError('tol must be a number')
        if tol < 0:
            raise ValueError('tol must be greater than or equal to 0')

        # Check that max_iter is of the correct type and within the expected range
        if not isinstance(max_iter, int):
            raise TypeError('max_iter must be an integer')
        if max_iter <= 0:
            raise ValueError('max_iter must be greater than 0')
        
        # Initialize instance variables
        self.k = k
        self.tol = float(tol)
        self.max_iter = max_iter
        self.centroids = None
        self.mse = -1.0
        
    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """

        # Check that input matrix is of the correct type, shape, and contents 
        if not isinstance(mat, np.ndarray):
            raise TypeError('Input matrix must be a numpy array')
        if mat.size == 0:
            raise ValueError('Input matrix must be non-empty')
        if mat.ndim != 2:
            raise ValueError('Input matrix must be 2-dimensional')
        if mat.shape[0] < self.k:
            raise ValueError('Input matrix must have at least k rows/observations')

        # 1. Initialize k centroids from k random data points (without replacement)
        self.centroids = mat[np.random.choice(mat.shape[0], self.k, replace=False), :]

        # 1.5. Iterate until the max centroid change is within tolerance or max iterations reached
        for _ in range(self.max_iter):

            # 2. For each data point, compute the distance to each centroid, and find the closest centroid
            dists = cdist(self.centroids, mat, metric='euclidean')
            classifications = np.argmin(dists, axis=0)

            # 3. Update the centroids to be the average of their closest data points found in (2)
            prev_centroids = self.centroids.copy()
            for i in range(self.k):
                self.centroids[i] = mat[classifications == i].mean(axis=0)

            # 4. Compute max change in a centroid from the previous centroid
            centroid_change = np.linalg.norm(self.centroids - prev_centroids, axis=1)
            max_change = np.max(centroid_change)

            # 5. Repeat (2) through (4) until the change in centroid is less than user-defined tolerance
            if max_change < self.tol:
                break
        
        # 6. Compute and store the mse between each data point and it's closest centroid
        dists = cdist(self.centroids, mat, metric='euclidean')
        min_dists = np.min(dists, axis=0)
        self.mse = np.mean(min_dists**2)

    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """

        # Check that input matrix is of the correct type, shape, and contents 
        if not isinstance(mat, np.ndarray):
            raise TypeError('Input matrix must be a numpy array')
        if mat.size == 0:
            raise ValueError('Input matrix must be non-empty')
        if mat.ndim != 2:
            raise ValueError('Input matrix must be 2-dimensional')
        if mat.shape[1] != self.centroids.shape[1]:
            raise ValueError(f'Input matrix must have the same number of columns/features as the model\'s centroids {self.centroids.shape[1]}')
        
        # Compute the distances from each data point to each centroid
        dists = cdist(self.centroids, mat, metric='euclidean')

        # For each data point, get the index of the centroid to which it is closest to
        # This index will be the cluster label for that data point
        classifications = np.argmin(dists, axis=0)
        return classifications

    def get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """
        return self.mse

    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        return self.centroids
