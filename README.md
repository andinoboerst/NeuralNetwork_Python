# Neural Network in Python with a OOP approach

My attempt at a simple neural network in python using OOP.
The network is made up of layers, consisting of nodes which are connected by edges.

## The Classes

### The Neural Network Class

This class contains all the parameters of the NN as well as a list with all of the layers in the NN.
It also includes the "fit" function to initialize the backpropagation to optimize the weights and biases.

### The Layer Class

It contains a list of all the nodes inside the layer as well as a pointer to the previous layer of the NN.
The are different layer types, for example input layer and output layer which all inherit from the base Layer class.

### The Node Class

The node class contains the information about the bias of the node as well as the result of the last prediction.
Additionally it stores the edges that connect to the node from the previous layer's nodes.

### The Edge Class

This is a very simple class only saving information about the weight of the edge and the nodes it connects to.


## Results

The NN is quite simple and is prone to getting stuck in local minima. Different activation functions are available, however more layer types and error functions could be added.
The example included in the `main.py` file is the sklearn digits dataset. It consits of handwritten numbers given in a (16x16) pixel image with pixel values ranging from 0 to 100.

The dataset is split in half meaning 898 samples are both in the training and the test set. After 50 iterations with stochastic gradient descent the NN can achieve an accuracy of more than 85% on the test set.


