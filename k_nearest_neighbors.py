# k_nearest_neighbors.py: Machine learning implementation of a K-Nearest Neighbors classifier from scratch.
#
# Submitted by: [enter your full name here] -- [enter your IU username here]
#
# Based on skeleton code by CSCI-B 551 Fall 2021 Course Staff

import numpy as np
from utils import euclidean_distance, manhattan_distance


class KNearestNeighbors:
    """
    A class representing the machine learning implementation of a K-Nearest Neighbors classifier from scratch.

    Attributes:
        n_neighbors
            An integer representing the number of neighbors a sample is compared with when predicting target class
            values.

        weights
            A string representing the weight function used when predicting target class values. The possible options are
            {'uniform', 'distance'}.

        _X
            A numpy array of shape (n_samples, n_features) representing the input data used when fitting the model and
            predicting target class values.

        _y
            A numpy array of shape (n_samples,) representing the true class values for each sample in the input data
            used when fitting the model and predicting target class values.

        _distance
            An attribute representing which distance metric is used to calculate distances between samples. This is set
            when creating the object to either the euclidean_distance or manhattan_distance functions defined in
            utils.py based on what argument is passed into the metric parameter of the class.

    Methods:
        fit(X, y)
            Fits the model to the provided data matrix X and targets y.

        predict(X)
            Predicts class target values for the given test data matrix X using the fitted classifier model.
    """

    def __init__(self, n_neighbors = 5, weights = 'uniform', metric = 'l2'):
        # Check if the provided arguments are valid
        if weights not in ['uniform', 'distance'] or metric not in ['l1', 'l2'] or not isinstance(n_neighbors, int):
            raise ValueError('The provided class parameter arguments are not recognized.')

        # Define and setup the attributes for the KNearestNeighbors model object
        self.n_neighbors = n_neighbors
        self.weights = weights
        self._X = None
        self._y = None
        self._distance = euclidean_distance if metric == 'l2' else manhattan_distance

    def fit(self, X, y):
        """
        Fits the model to the provided data matrix X and targets y.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the input data.
            y: A numpy array of shape (n_samples,) representing the true class values for each sample in the input data.

        Returns:
            None.
        """
        self._X = X
        self._y = y


        #raise NotImplementedError('This function must be implemented by the student.')

    def predict(self, X):
        """
        Predicts class target values for the given test data matrix X using the fitted classifier model.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the test data.

        Returns:
            A numpy array of shape (n_samples,) representing the predicted target class values for the given test data.
        """

        #raise NotImplementedError('This function must be implemented by the student.')

        total_distances = []
        for test_sample in X:
            distances = []
            for training_sample in self._X:
                distances.append(self._distance(test_sample,training_sample))
            total_distances.append(distances)
        #the idea to enumerate distances list to get indices was taken from https://towardsdatascience.com/k-nearest-neighbors-classification-from-scratch-with-numpy-cb222ecfeac1
        k_neighbors_list =[]
        final_dist=[]
        final_indices_list=[]
        for distance_list in total_distances:
            sorted_list = enumerate(distance_list)
            sorted_list_final = sorted(sorted_list,key=lambda x: x[1])
            #take only first five neighbors
            sorted_list_final = sorted_list_final[:self.n_neighbors]

            indices_list = [row[0] for row in sorted_list_final]
            k_neighbors_list = [row[1] for row in sorted_list_final]

            final_dist.append(k_neighbors_list)
            final_indices_list.append(indices_list)
        final_dist_np = np.array(final_dist)
        final_indices_np = np.array(final_indices_list)
        n_samples = X.shape[0]
        y_pred = np.zeros(n_samples)
        if self.weights == 'uniform':
            i=0
            for index in final_indices_np:

                temp = np.bincount(self._y[index])
                y_pred[i]=np.argmax(temp)
                i+=1
            #print("y_pred is",y_pred)
            return y_pred
        if self.weights == 'distance':
            y_pred = np.zeros(n_samples)
            classes = self.n_neighbors
            inverse_dist = 1/(final_dist_np+1)
            # print("final_indices are",final_indices_np)
            # print("final distances are",final_dist_np)
            labels = np.array([self._y[index] for index in final_indices_np])
            # print("labels are",labels)
            #count_dict = {key:0 for key in range(classes)}

            for i in range(labels.shape[0]):
                classes_count = np.zeros(np.max(self._y)+1)
                for j in range(len(classes_count)):
                    indices = np.where(labels[i]==j)
                    # print("i and j is",i,j)
                    # print("labels[i] was {} and class_count[j was {}".format(labels[i],classes_count[j]))
                    # print("indices are",indices)
                    classes_count[j] = np.sum(inverse_dist[i][indices])
                # print("classes counts are{} for labels[i] and ",classes_count)

                y_pred[i] = np.argmax(classes_count)
            return y_pred
















