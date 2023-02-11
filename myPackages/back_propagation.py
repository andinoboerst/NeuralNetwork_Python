import numpy as np
import math

def sample_prop(my_nn, X: np.array([[]]), Y: np.array([[]]), to_update: bool=True) -> None:
    '''
    Backpropagation after every sample
    '''
    for x, y in zip(X,Y):
        my_nn.predict([x])
        my_nn.layers[-1].define_errors(y)
        for layer in reversed(my_nn.layers[2:-1]):
            layer.define_errors()

        for layer in reversed(my_nn.layers[1:]):
            for node in layer.nodes:
                for edge in node.edges:
                    res = edge.weight-(my_nn.learning_rate*node.error*edge.nodes[0].result)
                    edge.new_weights.append(res)

        if to_update:
            apply_prop(my_nn)



def mini_prop(my_nn, X: np.array([[]]), Y: np.array([[]]), num_batches: int=5) -> None:
    '''
    Backpropagation after a mini-batch
    batch_ratio is the percentage of samples that should be used for each mini batch
    '''
    if type(num_batches) != int or num_batches < 1:
        raise ValueError("Given number of batches is not permitted.")
    # Divide the sets and call batch_prop() on them
    for batch_X, batch_Y in zip(np.array_split(X,num_batches),np.array_split(Y,num_batches)):
        batch_prop(my_nn, batch_X, batch_Y)


def batch_prop(my_nn, X: np.array([[]]), Y: np.array([[]])) -> None:
    '''
    Backpropagation for the entire batch
    '''
    # call sample_prop() and average the results
    sample_prop(my_nn, X,Y,to_update=False)
    apply_prop(my_nn)


def apply_prop(my_nn):
    '''
    Apply the defined weights and biases to the neural network
    '''
    for layer in reversed(my_nn.layers[1:]):
        for node in layer.nodes:
            for edge in node.edges:
                edge.weight = np.mean(edge.new_weights)
                edge.new_weights = []



def mean_squared_error(self, y_pred: np.array([]), y_real: np.array([])):
    return ((y_pred-y_real)**2).sum() / (2*len(y_pred))