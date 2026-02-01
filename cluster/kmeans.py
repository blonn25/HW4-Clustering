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

        # check that inputs are of correct type and are within the expected range
        if type(k) == int and k > 0:
            self.k = k
        else:
            raise TypeError('k must be an integer greater than 0')
        
        if type(tol) == float and tol >= 0:
            self.tol = tol
        else:
            raise TypeError('tol must be a float greater than or equal to 0')
        
        if type(max_iter) == int and max_iter > 0:
            self.max_iter = max_iter
        else:
            raise TypeError('max_iter must be an integer greater than 0')
        
        # initialize empty centroids and data_pts parameters
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

        # check that input matrix is of the correct type and shape 
        # if so, select k random centroids from the data points
        if type(mat) == np.ndarray and mat.ndim == 2 and mat.size > 0:
            self.centroids = mat[np.random.choice(mat.shape[0], self.k, replace=False), :]
        else:
            raise TypeError('Input matrix must be a non-empty 2D numpy array where the' \
                            'rows are observations and columns are features')        

        # 1.5. Iterate until convergence or max iterations reached.
        for i in range(self.max_iter):

            # 2. For each data point, compute the distance to each centroid, and find the closest centroid.
            dists = cdist(self.centroids, mat, metric='euclidean')
            classifications = np.argmin(dists, axis=0)

            # 3. Update the centroids to be the average of their closest data points found in (2).
            prev_centroids = self.centroids.copy()
            for idx in range(self.k):
                self.centroids[idx] = mat[classifications == idx].mean(axis=0)

            # 4. Compute max change in a centroid from the previous centroid.
            centroid_change = np.linalg.norm(self.centroids - prev_centroids, axis=1)
            max_change = np.max(centroid_change)

            # 5. Repeat (2) through (4) until the change in centroid is less than some epsilon.
            if max_change < self.tol:
                break
        
        # 6. Compute and store the mse between each data point and it's closest centroid.
        dists = cdist(self.centroids, mat, metric='euclidean')
        self.mse = np.mean(np.min(dists, axis=0)**2)
        

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

        # check that input matrix is of the correct type and shape 
        if type(mat) == np.ndarray and mat.ndim == 2 and mat.size > 0:
            pass
        else:
            raise TypeError('Input matrix must be a non-empty 2D numpy array where the' \
                            'rows are observations and columns are features')
        
        # compute the distances from each data point to each centroid
        dists = cdist(self.centroids, mat, metric='euclidean')

        # for each data point, get the index of the centroid to which it is closest to
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
