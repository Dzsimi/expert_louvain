# coding=utf-8


class Status(object):
    """
    To handle several data in one struct.

    Could be replaced by named tuple, but don't want to depend on python 2.6
    """
    node2com = {}
    total_weight = 0
    internals = {}
    degrees = {} # community degree dictionary (key: community index, value: degree sum of nodes in the particular community)
    gdegrees = {} # node degree dictionary (key: node index, value: degree of node)
    loops = {}
    expected_internal = {}
    expected_total_weight = 0

    def __init__(self):
        self.node2com = dict([])
        self.total_weight = 0
        self.degrees = dict([])
        self.gdegrees = dict([])
        self.internals = dict([])
        self.loops = dict([])
        self.expected_internal = dict([])
        self.expected_total_weight = 0

    def __str__(self):
        return ("node2com : " + str(self.node2com) + " degrees : "
                + str(self.degrees) + " internals : " + str(self.internals)
                + " total_weight : " + str(self.total_weight))

    def copy(self):
        """Perform a deep copy of status"""
        new_status = Status()
        new_status.node2com = self.node2com.copy()
        new_status.internals = self.internals.copy()
        new_status.degrees = self.degrees.copy()
        new_status.gdegrees = self.gdegrees.copy()
        new_status.total_weight = self.total_weight
        new_status.expected_internal = self.expected_internal.copy()
        new_status.expected_total_weight = self.expected_total_weight
        new_status.loops = self.loops.copy()

    def init(self, graph, weight, part=None, expected_matrix=None):
        """Initialize the status of a graph with every node in one community"""
        count = 0
        self.node2com = dict([])
        self.total_weight = 0
        self.degrees = dict([])
        self.gdegrees = dict([])
        self.internals = dict([])
        self.total_weight = graph.size(weight=weight)
        self.expected_internal = dict([])
        self.expected_total_weight = 0
        if expected_matrix is not None: # this part essentially populate the expected_internal dicitonary (expected spatially adjusted weights within community) which is used to calculate modularity
            self.expected_total_weight = expected_matrix.sum() / 2

            for node in graph.nodes():
                com = part[node] if part else count
                for neighbor in graph.nodes():
                    if part and part[neighbor] != com:
                        continue
                    if node == neighbor:
                        self.expected_internal[com] = self.expected_internal.get(com, 0.0) + expected_matrix[node, neighbor]
                    else:
                        self.expected_internal[com] = self.expected_internal.get(com, 0.0) + expected_matrix[node, neighbor] / 2.0
                if not part:
                    count += 1
        if part is None: # if no partition is created yet, it assigns all nodes to its own community and creates essential dictionaries
            count = 0
            for node in graph.nodes():
                self.node2com[node] = count
                deg = float(graph.degree(node, weight=weight))
                if deg < 0:
                    error = "Bad node degree ({})".format(deg)
                    raise ValueError(error)
                self.degrees[count] = deg
                self.gdegrees[node] = deg
                edge_data = graph.get_edge_data(node, node, default={weight: 0})
                self.loops[node] = float(edge_data.get(weight, 1))
                self.internals[count] = self.loops[node]
                count += 1
        else:
            for node in graph.nodes():
                com = part[node]
                self.node2com[node] = com
                deg = float(graph.degree(node, weight=weight))
                self.degrees[com] = self.degrees.get(com, 0) + deg
                self.gdegrees[node] = deg
                inc = 0.
                for neighbor, datas in graph[node].items():
                    edge_weight = datas.get(weight, 1)
                    if edge_weight <= 0:
                        error = "Bad graph type ({})".format(type(graph))
                        raise ValueError(error)
                    if part[neighbor] == com:
                        if neighbor == node:
                            inc += float(edge_weight)
                        else:
                            inc += float(edge_weight) / 2.
                self.internals[com] = self.internals.get(com, 0) + inc
                