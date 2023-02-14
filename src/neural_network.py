import src.layer as lay
import src.back_propagation as bp
import src.error_functions as errf
import numpy as np

LAYER_TYPES = {"standard": lay.Layer, "input": lay.Layer, "output": lay.Layer}


class NeuralNetwork:
    propagation_types = {"sample": bp.sample_prop, "mini-batch": bp.mini_prop, "batch": bp.batch_prop}

    def __init__(self, propagation: str="mini-batch", learning_rate: float=0.1, max_iterations: int=200) -> None:
        if propagation not in self.propagation_types:
            raise ValueError("Propagation type not supported.")
        if learning_rate <= 0:
            raise ValueError("Learning rate cannot 0 or negative.")
        if type(max_iterations) != int or  max_iterations < 1:
            raise ValueError("Max Iterations has to be positive integer.")
        
        self.propagation = self.propagation_types[propagation]
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.layers = []

    def check_input_layer(self) -> None:
        if len(self.layers) == 0:
            raise AttributeError("Need to add input layer first.")

    def add_input_layer(self, num_nodes: int) -> None:
        self.layers.append(lay.Layer(num_nodes))

    def add_standard_layer(self, num_nodes: int, act_func: str="relu", err_func: str="mse") -> None:
        self.check_input_layer()
        self.layers.append(lay.Layer(num_nodes, act_func, self.layers[-1], err_func))

    def add_output_layer(self, num_nodes: int, act_func: str="softmax", err_func: str="mse") -> None:
        self.check_input_layer()
        self.layers.append(lay.OutputLayer(num_nodes, act_func, self.layers[-1], err_func))

    def predict(self, input: np.array([[]])) -> list:
        if len(input[0]) != len(self.layers[0].nodes):
            raise IndexError("Input is not equivalent to size of input layer.")
        results = []
        for entry in input:
            for value, node in zip(entry, self.layers[0].nodes):
                node.result = value
            for layer in self.layers[1:]:
                layer.layer_predict()
            results.append([node.result for node in self.layers[-1].nodes])
        return np.array(results)
    
    def fit(self, X: list, Y: list) -> None:
        for i in range(self.max_iterations):
            print(f"Iteration {i+1} of maximum {self.max_iterations}.")
            self.propagation(self, X, Y)
            Y_predicted = self.predict(X)
            loss = errf.mean_squared_error(Y_predicted, Y)
            acc = sum(np.argmax(Y_predicted, axis=1)==np.argmax(Y, axis=1))/len(Y)*100
            print(f"Loss: {loss}.")
            print(f"Accuracy: {acc:.2f}%.\n")
            if acc > 99:
                break

        

