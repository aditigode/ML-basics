# utils.py: Utility file for implementing helpful utility functions used by the ML algorithms.
#
# Submitted by: [enter your full name here] -- [enter your IU username here]
#
# Based on skeleton code by CSCI-B 551 Fall 2021 Course Staff

import numpy as np


def euclidean_distance(x1, x2):
    """
    Computes and returns the Euclidean distance between two vectors.

    Args:
        x1: A numpy array of shape (n_features,).
        x2: A numpy array of shape (n_features,).
    """

    #raise NotImplementedError('This function must be implemented by the student.')

    return np.sqrt(np.sum(np.square(x1-x2)))


def manhattan_distance(x1, x2):
    """
    Computes and returns the Manhattan distance between two vectors.

    Args:
        x1: A numpy array of shape (n_features,).
        x2: A numpy array of shape (n_features,).
    """
    manhattan_dist=0
    for i in range(x1.shape[0]):
        manhattan_dist += abs(x1[i]-x2[i])
    return manhattan_dist

    #raise NotImplementedError('This function must be implemented by the student.')


def identity(x, derivative = False):
    """
    Computes and returns the identity activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """
    if not derivative:
        return x
    else:
        return 1
    #raise NotImplementedError('This function must be implemented by the student.')


def sigmoid(x, derivative = False):
    """
    Computes and returns the sigmoid (logistic) activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """
    #result = np.exp(x)/ sum(np.exp(Z))
    if not derivative:
        result = 1/ (1 + np.exp(-x))
    #raise NotImplementedError('This function must be implemented by the student.')
        return result
    else:
        result=sigmoid(x)* (1-sigmoid(x))
        return result


def tanh(x, derivative = False):
    """
    Computes and returns the hyperbolic tangent activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """
    if not derivative:
        # value= 2 * x
        # result = (2/(1 + np.exp(-value))) - 1

        return np.tanh(x)
    else:
        result = 1 - (tanh(x) * tanh(x))
        return result

    #raise NotImplementedError('This function must be implemented by the student.')


def relu(x, derivative = False):
    """
    Computes and returns the rectified linear unit activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """
    if not derivative:
        return np.maximum(0.0,x)
    else:
        return np.where(x>=0, 1,0)
    #raise NotImplementedError('This function must be implemented by the student.')


def softmax(x, derivative = False):
    x = np.clip(x, -1e100, 1e100)
    if not derivative:
        c = np.max(x, axis = 1, keepdims = True)
        return (np.exp(x - c - np.log(np.sum(np.exp(x - c), axis = 1, keepdims = True))))
    else:
        return (softmax(x) * (1 - softmax(x)))


def cross_entropy(y, p):
    """
    Computes and returns the cross-entropy loss, defined as the negative log-likelihood of a logistic model that returns
    p probabilities for its true class labels y.

    Args:
        y:
            A numpy array of shape (n_samples, n_outputs) representing the one-hot encoded target class values for the
            input data used when fitting the model.
        p:
            A numpy array of shape (n_samples, n_outputs) representing the predicted probabilities from the softmax
            output activation function.
    """
    n_outputs, n_samples = y.shape
    #loss = np.zeros((n_samples))
    # y=y.T
    # p=p.T
    #
    # for i in range(n_samples):
    #     index=np.argmax(y[i])
    #     # print("index is",index)
    #     # print("y is",y)
    #     # print("y[i][index]",y[i][index])
    #     # print("p is",p)
    #     #print("np.log(p[i][index]",np.log(p[i][index]))
    #     #print(loss)
    #     #loss[i] = -y[i][index] * np.log(p[i][index])
    #     loss[i]= -np.sum(y[i]* np.log(p[i]))
    #     #print(loss[i])
    #the idea of clipping predicted labels to avoid log values to have infinity values was taken from https://www.python-engineer.com/courses/pytorchbeginner/11-softmax-and-crossentropy/
    limit = 1e-15
    p = np.clip(p, limit, 1 - limit)
    loss = -np.sum(y * np.log(p))
    return loss
    # #loss = np.sum(loss)
    # #print("loss is",loss)
    # return loss
    #pass
    # for j in range(n_outputs):
    #raise NotImplementedError('This function must be implemented by the student.')


def one_hot_encoding(y):
    """
    Converts a vector y of categorical target class values into a one-hot numeric array using one-hot encoding: one-hot
    encoding creates new binary-valued columns, each of which indicate the presence of each possible value from the
    original data.

    Args:
        y: A numpy array of shape (n_samples,) representing the target class values for each sample in the input data.

    Returns:
        A numpy array of shape (n_samples, n_outputs) representing the one-hot encoded target class values for the input
        data. n_outputs is equal to the number of unique categorical class values in the numpy array y.
    """
    one_hot_encoded_y = np.zeros((y.shape[0],np.max(y)+1))
    for i in range(y.size):
        one_hot_encoded_y[i,y[i]] = 1
    return one_hot_encoded_y.T



    #raise NotImplementedError('This function must be implemented by the student.')
