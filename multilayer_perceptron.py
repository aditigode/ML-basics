# multilayer_perceptron.py: Machine learning implementation of a Multilayer Perceptron classifier from scratch.
#
# Submitted by: [enter your full name here] -- [enter your IU username here]
#
# Based on skeleton code by CSCI-B 551 Fall 2021 Course Staff

import numpy as np
from utils import identity, sigmoid, tanh, relu, softmax, cross_entropy, one_hot_encoding
#https://medium.com/swlh/weight-initialization-technique-in-neural-networks-fc3cbcd03046

class MultilayerPerceptron:
    """
    A class representing the machine learning implementation of a Multilayer Perceptron classifier from scratch.

    Attributes:
        n_hidden
            An integer representing the number of neurons in the one hidden layer of the neural network.

        hidden_activation
            A string representing the activation function of the hidden layer. The possible options are
            {'identity', 'sigmoid', 'tanh', 'relu'}.

        n_iterations
            An integer representing the number of gradient descent iterations performed by the fit(X, y) method.

        learning_rate
            A float representing the learning rate used when updating neural network weights during gradient descent.

        _output_activation
            An attribute representing the activation function of the output layer. This is set to the softmax function
            defined in utils.py.

        _loss_function
            An attribute representing the loss function used to compute the loss for each iteration. This is set to the
            cross_entropy function defined in utils.py.

        _loss_history
            A Python list of floats representing the history of the loss function for every 20 iterations that the
            algorithm runs for. The first index of the list is the loss function computed at iteration 0, the second
            index is the loss function computed at iteration 20, and so on and so forth. Once all the iterations are
            complete, the _loss_history list should have length n_iterations / 20.

        _X
            A numpy array of shape (n_samples, n_features) representing the input data used when fitting the model. This
            is set in the _initialize(X, y) method.

        _y
            A numpy array of shape (n_samples, n_outputs) representing the one-hot encoded target class values for the
            input data used when fitting the model.

        _h_weights
            A numpy array of shape (n_features, n_hidden) representing the weights applied between the input layer
            features and the hidden layer neurons.

        _h_bias
            A numpy array of shape (1, n_hidden) representing the weights applied between the input layer bias term
            and the hidden layer neurons.

        _o_weights
            A numpy array of shape (n_hidden, n_outputs) representing the weights applied between the hidden layer
            neurons and the output layer neurons.

        _o_bias
            A numpy array of shape (1, n_outputs) representing the weights applied between the hidden layer bias term
            neuron and the output layer neurons.

    Methods:
        _initialize(X, y)
            Function called at the beginning of fit(X, y) that performs one-hot encoding for the target class values and
            initializes the neural network weights (_h_weights, _h_bias, _o_weights, and _o_bias).

        fit(X, y)
            Fits the model to the provided data matrix X and targets y.

        predict(X)
            Predicts class target values for the given test data matrix X using the fitted classifier model.
    """

    def __init__(self, n_hidden = 16, hidden_activation = 'sigmoid', n_iterations = 1000, learning_rate = 0.01):
        # Create a dictionary linking the hidden_activation strings to the functions defined in utils.py
        activation_functions = {'identity': identity, 'sigmoid': sigmoid, 'tanh': tanh, 'relu': relu}

        # Check if the provided arguments are valid
        if not isinstance(n_hidden, int) \
                or hidden_activation not in activation_functions \
                or not isinstance(n_iterations, int) \
                or not isinstance(learning_rate, float):
            raise ValueError('The provided class parameter arguments are not recognized.')

        # Define and setup the attributes for the MultilayerPerceptron model object
        self.n_hidden = n_hidden
        self.hidden_activation = activation_functions[hidden_activation]
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self._output_activation = softmax
        self._loss_function = cross_entropy
        self._loss_history = []
        self._X = None
        self._y = None
        self._h_weights = None
        self._h_bias = None
        self._o_weights = None
        self._o_bias = None

    def _initialize(self, X, y):
        """
        Function called at the beginning of fit(X, y) that performs one hot encoding for the target class values and
        initializes the neural network weights (_h_weights, _h_bias, _o_weights, and _o_bias).

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the input data.
            y: A numpy array of shape (n_samples,) representing the true class values for each sample in the input data.

        Returns:
            None.
        """
        np.random.seed(42)
        self._X = X.T
        self._y = one_hot_encoding(y)
        n_samples = self._X.shape[1]
        n_features = self._X.shape[0]

        n_outputs = self._y.shape[0]
        #weight initialization was referred from https://medium.com/swlh/weight-initialization-technique-in-neural-networks-fc3cbcd03046
        if self.hidden_activation == 'relu':
            #He initialization
            self._h_weights = np.random.randn(self.n_hidden,n_features) * np.sqrt(2/self.n_features)
        elif self.hidden_activation == 'tanh' or self.hidden_activation == 'sigmoid':
            #xavier initialization
            self._h_weights = np.random.randn(self.n_hidden, n_features) * np.sqrt(2/ (n_features+self.n_hidden))
        else:
            self._h_weights = np.random.randn(self.n_hidden, n_features) * 0.1

        self._h_bias = np.random.randn(self.n_hidden,1)

        self._o_weights = np.random.randn(n_outputs,self.n_hidden) * np.sqrt(2/ (self.n_hidden+n_outputs))
        self._o_bias = np.random.randn(n_outputs, 1)





        #raise NotImplementedError('This function must be implemented by the student.')

    def fit(self, X, y):
        """
        Fits the model to the provided data matrix X and targets y and stores the cross-entropy loss every 20
        iterations.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the input data.
            y: A numpy array of shape (n_samples,) representing the true class values for each sample in the input data.

        Returns:
            None.
        """

        self._initialize(X, y)
        #feed-forward
        #input to hidden layer
        # print(" x shape is",self._X.shape)
        # print("y shape is",self._y.shape)
        # print("h weights are",self._h_weights.shape)
        # print("h bias is",self._h_bias.shape)
        # print("o weights are", self._o_weights.shape)
        # print("o bias is", self._o_bias.shape)
        for i in range(self.n_iterations):
            Z1 = np.dot(self._h_weights,self._X) + self._h_bias

            A1 = self.hidden_activation(Z1.T).T

            #hidden to output layer
            Z2 = (np.dot(self._o_weights,A1) )+ self._o_bias
            A2 = self._output_activation(Z2.T).T


            #back-propogation
            #y should n_outputs X n_samples because A2 is n_output X n_samples
            dZ2 = A2 - self._y
            n_samples = self._X.shape[1]
            dW2 = (1/n_samples) * np.dot(dZ2,A1.T)
            db2 = (1/n_samples) * np.sum(dZ2,axis=1,keepdims=True)

            temp = (self.hidden_activation(Z1.T, True))
            if type(temp)==int:
                dZ1 = np.dot(self._o_weights.T, dZ2) * temp
            else:
                dZ1 = np.dot(self._o_weights.T, dZ2) * temp.T

            dW1 = (1/n_samples) * np.dot(dZ1,self._X.T)
            db1 = (1/n_samples) * np.sum(dZ1,axis=1,keepdims=True)
            #back-prop ends here

            #update parameters
            self._h_weights = self._h_weights - self.learning_rate * dW1
            self._h_bias = self._h_bias - self.learning_rate * db1
            self._o_weights = self._o_weights - self.learning_rate * dW2
            self._o_bias = self._o_bias - self.learning_rate * db2
            #calculate loss every 20 iterations
            if i % 20:
                loss = self._loss_function(self._y.T,A2.T)
                self._loss_history.append(loss)

        #raise NotImplementedError('This function must be implemented by the student.')
        #print("loss history is",self._loss_history)

    def predict(self, X):
        """
        Predicts class target values for the given test data matrix X using the fitted classifier model.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the test data.

        Returns:
            A numpy array of shape (n_samples,) representing the predicted target class values for the given test data.
        """

        #raise NotImplementedError('This function must be implemented by the student.')
        #feed-forward
        #input to hidden layer
        Z1 = np.dot(self._h_weights, X.T) + self._h_bias
        A1 = self.hidden_activation(Z1.T).T
        # hidden to output layer
        Z2 = np.dot(self._o_weights, A1) + self._o_bias
        A2 = self._output_activation(Z2.T).T
        # print("A2 is",A2)
        #find predicted labels using max value from the three classes
        output = np.argmax(A2,axis=0)

        return output.T

