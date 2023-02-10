import myPackages.activation_functions as af
import myPackages.back_propagation as bp
import myPackages.graph_viz as gv

LAYER_TYPES = {"standard", "input", "output"}
ACTIVATION_FUNCTIONS = {"linear": af.linear_activation, "hyperbolic": af.hyperbolic_activation}


class NeuralNetwork:
    propagation_types = {"sample": bp.sample_prop, "mini-batch": bp.mini_prop, "batch": bp.batch_prop}

    def __init__(self, propagation: str="sample", learning_rate: float=0.1) -> None:
        if propagation not in self.propagation_types:
            raise ValueError("Propagation type not supported.")
        self.propagation = self.propagation_types[propagation]
        if learning_rate <= 0:
            raise ValueError("Learning rate cannot 0 or negative.")
        self.learning_rate = learning_rate
        self.layers = []

    def add_layer(self, num_nodes: int, nn_type: str="standard", act_func: str="linear") -> None:
        '''
        Add a layer to the Neural Network
        inputs: 
            num of nodes in the layer
            type of the layer (default: standard, other options: input, output)
        '''
        if nn_type not in LAYER_TYPES:
            raise TypeError("Layer Type not supported.")
        elif num_nodes < 1 or type(num_nodes) is not int:
            raise ValueError("Number of nodes must be a positive integer.")
        elif len(self.layers) == 0 and nn_type != "input":
            raise AttributeError("Need to add input layer first.")
        elif nn_type == "input":
            self.layers.append(Layer(num_nodes, nn_type, act_func))
        else:
            self.layers.append(Layer(num_nodes, nn_type, act_func, self.layers[-1]))

    def predict(self, input: list) -> list:
        if len(input) != len(self.layers[0].nodes):
            raise IndexError("Input is not equivalent to size of input layer.")
        for value, node in zip(input, self.layers[0].nodes):
            node.result = value
        for layer in self.layers[1:]:
            layer.layer_predict()
        return [node.result for node in self.layers[-1].nodes]
    
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

    

class Layer:
    def __init__(self, num_nodes: int, layer_type: str, act_func: str, prev_layer=None) -> None:
        self.num_nodes = num_nodes
        self.layer_type = layer_type
        self.activation_function = ACTIVATION_FUNCTIONS[act_func]
        self.prev_layer = prev_layer
        if self.prev_layer is not None:
            self.nodes = [Node(self.prev_layer.num_nodes) for i in range(self.num_nodes)]
        else:
            self.nodes = [Node() for i in range(self.num_nodes)]

    def layer_predict(self) -> None:
        for node in self.nodes:
            res = node.bias
            for w, prev_node in zip(node.weights,self.prev_layer.nodes):
                res += w*prev_node.result
            node.store_result(self.activation_function(res))


class Node:
    def __init__(self, num_w: int=0) -> None:
        self.bias = 0
        self.weights = [1 for i in range(num_w)]

    def store_result(self, res: float) -> None:
        self.result = res
        

