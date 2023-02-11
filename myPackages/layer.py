import myPackages.activation_functions as af
import myPackages.node as nd
import myPackages.edge as ed
import random

ACTIVATION_FUNCTIONS = {"linear": af.linear_activation, "hyperbolic": af.hyperbolic_activation, "sigmoid": af.sigmoid_activation, "relu": af.relu_activation, "prelu": af.prelu_activation, "elu": af.elu_activation}
DERIV_ACTIVATION_FUNCTIONS = {"linear": af.deriv_linear, "hyperbolic": af.deriv_hyperbolic, "sigmoid": af.deriv_sigmoid, "relu": af.deriv_relu, "prelu": af.deriv_prelu, "elu": af.deriv_elu}


class Layer:
    def __init__(self, num_nodes: int, act_func: str="linear", prev_layer=None) -> None:
        if num_nodes < 1 or type(num_nodes) is not int:
            raise ValueError("Number of nodes must be a positive integer.")
        self.num_nodes = num_nodes
        self.activation_function = ACTIVATION_FUNCTIONS[act_func]
        self.deriv_activation = DERIV_ACTIVATION_FUNCTIONS[act_func]
        self.prev_layer = prev_layer
        self.initialize_nodes_edges()

    def initialize_nodes_edges(self):
        self.nodes = [nd.Node() for i in range(self.num_nodes)]
        if self.prev_layer is not None:
            for node2 in self.nodes:
                node2.add_edges([ed.Edge(node1, node2, random.uniform(0,1)) for node1 in self.prev_layer.nodes])
            

    def layer_predict(self) -> None:
        for node in self.nodes:
            res = node.bias
            for edge in node.edges:
                res += edge.weight*edge.nodes[0].result
            node.store_result(self.activation_function(res))

    def define_errors(self):
        for node in self.nodes:
            for edge in node.edges:
                edge.nodes[0].error = (node.error*edge.weight)*self.deriv_activation(node.result)


class OutputLayer(Layer):
    def __init__(self, num_nodes: int, act_func: str="linear", prev_layer=None) -> None:
        super().__init__(num_nodes, act_func, prev_layer)

    def define_errors(self, y_expected):
        for result_expected, node in zip(y_expected, self.nodes):
            for edge in node.edges:
                node.error = (node.result - result_expected)*self.deriv_activation(node.result)
                edge.nodes[0].error = (node.error*edge.weight)*self.deriv_activation(node.result)


class ConvolutionalLayer(Layer):
    def __init__(self, num_nodes: int, act_func: str="linear", prev_layer=None, **kwargs) -> None:
        super().__init__(num_nodes, act_func, prev_layer)