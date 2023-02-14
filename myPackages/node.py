import myPackages.edge as ed

class Node:
    def __init__(self) -> None:
        self.bias = 0
        self.edges = []
        self.dzs = []

    def store_result(self, res: float) -> None:
        self.result = res

    def add_edges(self, edges: type[ed.Edge]):
        self.edges.extend(edges)