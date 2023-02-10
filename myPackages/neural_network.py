import myPackages.layer as lay
import myPackages.back_propagation as bp
import myPackages.graph_viz as gv
import numpy as np

LAYER_TYPES = {"standard": lay.Layer, "input": lay.Layer, "output": lay.Layer}


class NeuralNetwork:
    propagation_types = {"sample": bp.sample_prop, "mini-batch": bp.mini_prop, "batch": bp.batch_prop}

    def __init__(self, propagation: str="sample", learning_rate: float=0.1, max_iterations: int=100) -> None:
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

    def add_layer(self, num_nodes: int, layer_type: str="standard", act_func: str="hyperbolic", **kwargs) -> None:
        '''
        Add a layer to the Neural Network
        inputs: 
            num of nodes in the layer
            type of the layer (default: standard, other options: input, output)
        '''
        if layer_type not in LAYER_TYPES:
            raise TypeError("Layer Type not supported.")
        elif num_nodes < 1 or type(num_nodes) is not int:
            raise ValueError("Number of nodes must be a positive integer.")
        elif len(self.layers) == 0 and layer_type != "input":
            raise AttributeError("Need to add input layer first.")
        elif layer_type == "input":
            self.layers.append(LAYER_TYPES[layer_type](num_nodes, layer_type, act_func))
        else:
            self.layers.append(LAYER_TYPES[layer_type](num_nodes, layer_type, act_func, self.layers[-1], **kwargs))

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
    
    def fit(self, X: list, y: list) -> None:
        self.propagation(X, y)

    # def visualize(self):
    #     G = gv.GraphVisualization()
    #     node = 0
    #     for layer1, layer2 in zip(self.layers[:-1], self.layers[1:]):
    #         for node in self.layer
    #     G.addEdge(0, 2)
    #     G.addEdge(1, 2)
    #     G.addEdge(1, 3)
    #     G.addEdge(5, 3)
    #     G.addEdge(3, 4)
    #     G.addEdge(1, 0)
    #     G.visualize()
        

