class Edge:
    def __init__(self, node1, node2, initial_weight=0.5) -> None:
        self.weight = initial_weight
        self.nodes = (node1, node2)
        self.new_weights = []