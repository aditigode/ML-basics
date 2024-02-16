### Part 1:  K-Nearest Neighbors Classification
K-Nearest Neighbors is used for both classification and regression problems. Here, K-nearest neighbor is used for classification on the Iris and Digits dataset.    
In KNN, the label of unseen data point is predicted by finding the closest K training data points to it and then taking majority votes of those K neighboring points.   
The following functions were implemented:   
**euclidean_distance(x1, x2) in utils.py:** The Euclidean distance was found using numpy functions to first square the different between the testing and training data points, sum them up and then take a square root of it.         
**manhattan_distance(x1, x2) in utils.py:** The block distance was computed by taking the absolute different between training and test points and then summing them up to return.     
**fit(X, y) in k_nearest_neighbors.py:** Since KNN is a lazy learner and uses memory-based approach where it just memorizes the training set rather than learning a discriminative function for prediction, this function involved only copying and storing the training data and labels in the KNN class.      
**predict(X) in k_nearest_neighbors.py:** For making predictions, distance was calculated of the testing data from all the training data(distance metrics used were Manhattan and Euclidean). After this, K closest points(having the least distance) were chosen to vote for class for the test data points. This is the 'uniform' weight metric method of KNN.   
The other weight metric used was 'distance'. The idea here is that the data points closest to the unseen data point will carry more weight than a training data point farther from the unseen point. Inverse of distance is taken and assigned as weight to each K top neigboring points. Then, a weighted sum is taken of all the points that belong to the same class. The class having the highest weight wins and that class is used to label the unseen data point.   
My model for KNN scored almost the same as sklearn for both the datasets. The two datasets run in about 5 minutes.    


### Part 2: Multilayer Perceptron Classification   
The multilayer perceptron network in this assignment has exactly three layers: Input layer, hidden layer, and output layer.
The number of nodes in each of these layers vary according to the dataset and passed arguments. Since the target values are categorical in nature, and neural networks require data that is numerical, I used one-hot encoding to convert each categorical value into a different column. Three new columns were added for Iris dataset and 10 new columns were added for Digits dataset.    
After preparing the data, I initialized the weights and biases for input-to-hidden and hidden-to-output layers. Since He initialization worked really well for ReLu to achieve convergence soon and Xavier initialization worked really well for tanh, sigmoid and softmax activation functions, I initialized the weights according to different activation fucntions.    
The reason for this was that a normal random initializations of weights leads to problems of vanishing gradient if weight values are very low, and exploding gradient if weight values are very high. The idea is to keep the mean of the activations zero and variance of the activations same across every layer. He Normal and Xavier initializations exactly do that.    
**He Normal Initialization:** This is done by multiplying the randomly initialized weight matrix by square root of (2/size_of_previous_layer). This aims at bringing the variance of outputs to one.   
**Xavier Initialization:** When the network is tanh, sigmoid or softmax activated, Xavier initialization is used. Weights are initialized by multiplying the randomly initialized weights by square root of  (2/size_of_previous_layer + size_of_next_layer). 
Reference: https://medium.com/swlh/weight-initialization-technique-in-neural-networks-fc3cbcd03046
Neural nets algorithm given below was taken from Andrew Ng's lecture: https://www.youtube.com/watch?v=7bLEWDZng_M   

The transpose of training data is taken to make the calculations easier.
dimensions(X.T) = n_features X n_samples
X= X.T
Y= Y.T
dimensions(hidden_weights)= n_hidden X n_features
dimensions(hidden_bias) = n_hidden X 1
dimensions(output_weights)= n_outputs X n_hidden
dimensions(output_bias) = n_outputs X 1

The feed-forward is done using the following steps:    
input to hidden layer:   
Z1 = hidden_weights * X + hidden_bias    
A1 = hidden_activation(Z1) #pass to the hidden activation fucntion used    

hidden to output layer:   
Z2 = output_weights * A1 + output_bias    
A2 = output_activation(Z2)     

back-propagation is done by:     
dZ2 = A2 - Y     
dW2 = 1/n_samples * dZ2 * A1.T    
db2 = 1/n_samples * summation(dZ2)     

dZ1 = output_weights.T * dZ2 * derivate_of_hidden_activation(Z1)    
dW1 = 1/n_samples * dZ1 * X.T   
db1 = 1/n_samples * summation(dZ1)    

once the back-propagation is done, paramters are nudged using these numbers:     
hidden_weights = hidden_weights - learning_rate * dW1    
hidden_bias = hidden_bias - learning_rate * db1    
output_weights = output_weights - learning_rate * dW2    
output_bias = output_bias - learning_rate * db2     


Cross entropy for multi-class/categorical classes is calculated by:    
loss = - summation( actual_labels * log(predicted_labels))     
The loss values decreases as the errors in predicting right label reduce during the training time.    

### Challenges     
I am getting very low accuracy for learning rate 0.001, slightly better accuracy for learning rate 0.01(comparable to sklearn after more than 1000 iterations) and the best accuracy(almost comparable to sklearn for all iterations). This can be because 0.001 is taking very slow steps on a gradient descent towards a local minima. Taking such a low learning rate would require more number of iterations to converge. I used He and Xavier weight initializations to improve the accuracy by 5% but it's not comparable to sklearn for low learning rates.    











