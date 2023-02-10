import myPackages.activation_functions as af
import myPackages.node as nd
import myPackages.edge as ed

ACTIVATION_FUNCTIONS = {"linear": af.linear_activation, "hyperbolic": af.hyperbolic_activation}


class Layer:
    def __init__(self, num_nodes: int, layer_type: str, act_func: str, prev_layer=None) -> None:
        self.num_nodes = num_nodes
        self.layer_type = layer_type
        self.activation_function = ACTIVATION_FUNCTIONS[act_func]
        self.prev_layer = prev_layer
        self.initialize_nodes_edges()

    def initialize_nodes_edges(self):
        self.nodes = [nd.Node() for i in range(self.num_nodes)]
        if self.prev_layer is not None:
            for node2 in self.nodes:
                node2.add_edges([ed.Edge(node1, node2) for node1 in self.prev_layer.nodes])
            

    def layer_predict(self) -> None:
        for node in self.nodes:
            res = node.bias
            for edge in node.edges:
                res += edge.weight*edge.nodes[0].result
            node.store_result(self.activation_function(res))

    def connect_graph(self):
        print("do smth here")