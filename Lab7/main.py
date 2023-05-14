import matplotlib.pyplot as plt
import random
import time

# Node class, represents a single node in the network.
class Node:
     # Each node has an id and a dictionary to store connected nodes (edges) and their weights.
    def __init__(self, id):
        self.id = id
        self.edges = {}
# Method to add a connection between this node and another
    def add_edge(self, node, weight):
        self.edges[node] = weight

    def get_edges(self):
        return self.edges


class Network:
    def __init__(self, size):
        self.nodes = [Node(i) for i in range(size)]

    def add_edge(self, u, v, weight):
        self.nodes[u].add_edge(v, weight)
        self.nodes[v].add_edge(u, weight)

    def get_size(self):
        return len(self.nodes)

# Kruskal's algorithm for finding the minimum spanning tree of a network.
def kruskal(network):
    # A set to store the minimum spanning tree.
    mst = set()
    edges = [(u, v, w) for u in range(network.get_size()) for v, w in network.nodes[u].get_edges().items()]
    edges.sort(key=lambda x: x[2])
    # Initialize disjoint sets for each node.
    disjoint_sets = [{i} for i in range(network.get_size())]
    # Initialize disjoint sets for each node.
    for u, v, w in edges:
        for s in disjoint_sets:
            if u in s:
                set_u = s
            if v in s:
                set_v = s
        if set_u != set_v:
            mst.add((u, v))
            disjoint_sets.remove(set_u)
            disjoint_sets.remove(set_v)
            disjoint_sets.append(set_u.union(set_v))
    return mst

# Prim's algorithm for finding the minimum spanning tree of a network.
def prim(network):
    # A set to store the minimum spanning tree.
    mst = set()
    # Set of visited nodes, initially containing just the first node.
    visited = {0}
    # While there are unvisited nodes.
    while len(visited) < network.get_size():
        # Get all edges that connect a visited node with an unvisited node.
        edges = [(u, v, network.nodes[u].get_edges()[v]) for u in visited for v in network.nodes[u].get_edges() if v not in visited]
        # Choose the edge with the smallest weight.
        u, v, w = min(edges, key=lambda x: x[2])
        # Add the chosen edge to the MST and the new node to the set of visited nodes.
        mst.add((u, v))
        visited.add(v)
    return mst


# Function to generate networks and benchmark the Kruskal and Prim algorithms.
def benchmark(num_vertices_list):
    # Generate a list of networks of different sizes.
    network_list = [Network(num_vertices) for num_vertices in num_vertices_list]
    for network in network_list:
        # Add random edges between the nodes of the network.
        for u in range(network.get_size()):
            for v in range(u + 1, network.get_size()):
                network.add_edge(u, v, random.randint(1, 100))

    kruskal_times = benchmark_algorithm(kruskal, network_list)
    prim_times = benchmark_algorithm(prim, network_list)

    plt.plot(num_vertices_list, kruskal_times, label='Kruskal')
    plt.plot(num_vertices_list, prim_times, label='Prim')
    plt.xlabel('Number of vertices')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.show()


def benchmark_algorithm(algorithm, network_list):
    times = []
    for network in network_list:
        start_time = time.time()
        algorithm(network)
        end_time = time.time()
        times.append(end_time - start_time)
    return times


num_vertices_list = [10, 50, 100, 150, 200, 250, 300, 450, 500, 550,]
benchmark(num_vertices_list)
