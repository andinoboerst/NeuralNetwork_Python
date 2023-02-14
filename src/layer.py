import src.activation_functions as af
import src.node as nd
import src.edge as ed
import random
import numpy as np

ACTIVATION_FUNCTIONS = {"linear": af.linear_activation, "hyperbolic": af.hyperbolic_activation, "sigmoid": af.sigmoid_activation, "relu": af.relu_activation, "prelu": af.prelu_activation, "elu": af.elu_activation, "softmax": af.softmax_activation}
DERIV_ACTIVATION_FUNCTIONS = {"linear": af.deriv_linear, "hyperbolic": af.deriv_hyperbolic, "sigmoid": af.deriv_sigmoid, "relu": af.deriv_relu, "prelu": af.deriv_prelu, "elu": af.deriv_elu, "softmax": af.deriv_sigmoid}
#ERROR_FUNCTIONS = {"mse": errf.mean_squared_error, "cross-entropy": errf.cross_entropy_error}


class Layer:
    def __init__(self, num_nodes: int, act_func: str="linear", prev_layer=None, err_func: str="mse") -> None:
        if num_nodes < 1 or type(num_nodes) is not int:
            raise ValueError("Number of nodes must be a positive integer.")
        if act_func not in ACTIVATION_FUNCTIONS:
            raise ValueError("Activation function does not exist.")
        # if err_func not in ERROR_FUNCTIONS:
        #     raise ValueError("Error function not supported.")
        self.num_nodes = num_nodes
        self.act_func_name = act_func
        self.activation_function = ACTIVATION_FUNCTIONS[self.act_func_name]
        self.deriv_activation = DERIV_ACTIVATION_FUNCTIONS[act_func]
        #self.error_func = ERROR_FUNCTIONS[err_func]
        self.prev_layer = prev_layer
        self.initialize_nodes_edges()
        if self.act_func_name == "softmax":
            self.layer_predict = self.layer_predict_special
        else:
            self.layer_predict = self.layer_predict_standard

    def initialize_nodes_edges(self):
        self.nodes = [nd.Node() for i in range(self.num_nodes)]
        if self.prev_layer is not None:
            for node2 in self.nodes:
                node2.add_edges([ed.Edge(node1, node2, random.uniform(-1,1)) for node1 in self.prev_layer.nodes])
            

    def layer_predict_standard(self) -> None:
        for node in self.nodes:
            res = node.bias
            for edge in node.edges:
                res += edge.weight*edge.nodes[0].result
            node.store_result(self.activation_function(res))

    def layer_predict_special(self) -> None:
        node_results = []
        for node in self.nodes:
            res = node.bias
            for edge in node.edges:
                res += edge.weight*edge.nodes[0].result
            node_results.append(res)
        new_node_results = self.activation_function(node_results)
        for node, result in zip(self.nodes,new_node_results):
            node.store_result(result)


    def define_nodal_gradient(self):
        for node in self.nodes:
            for edge in node.edges:
                edge.nodes[0].dzs.append((node.dz*edge.weight)*self.deriv_activation(edge.nodes[0].result))

        for node in self.prev_layer.nodes:
            node.dz = np.mean(node.dzs)
            node.dzs = []


class OutputLayer(Layer):
    def __init__(self, num_nodes: int, act_func: str="linear", prev_layer=None, err_func: str="mse") -> None:
        super().__init__(num_nodes, act_func, prev_layer, err_func)

    def define_nodal_gradient(self, errors):
        for error, node in zip(errors, self.nodes):
            node.dz = error*self.deriv_activation(node.result)

        super().define_nodal_gradient()


class ConvolutionalLayer(Layer):
    def __init__(self, num_nodes: int, act_func: str="linear", prev_layer=None, err_func: str="mse", **kwargs) -> None:
        super().__init__(num_nodes, act_func, prev_layer, err_func)