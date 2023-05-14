import numpy as np
import time
import networkx as nx
import matplotlib.pyplot as plt

class GraphAlgorithms:
    def __init__(self, graph):
        self.graph = graph
        self.n = len(graph)

    def dijkstra(self, start):
        dist = [float('inf')] * self.n
        visited = [False] * self.n
        dist[start] = 0
        for _ in range(self.n):
            u = min((x for x in range(self.n) if not visited[x]), key=dist.__getitem__)
            visited[u] = True
            for v in range(self.n):
                if self.graph[u][v] != 0 and not visited[v]:
                    alt = dist[u] + self.graph[u][v]
                    if alt < dist[v]:
                        dist[v] = alt
        return dist

    def floyd_warshall(self):
        dist = self.graph.copy()
        for k in range(self.n):
            for i in range(self.n):
                for j in range(self.n):
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
        return dist

    def draw_graph(self):
        G = nx.DiGraph()
        for i in range(self.n):
            for j in range(self.n):
                if self.graph[i][j] != 0:
                    G.add_edge(i, j, weight=self.graph[i][j])
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True)
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        plt.show()

def generate_graph(n, graph_type='sparse'):
    if graph_type == 'sparse':
        return np.random.choice([0, 1], size=(n, n), p=[0.9, 0.1])
    else:
        return np.random.randint(0,10, size=(n, n))

def measure_algorithm_times(algorithm, graph_type):
    times = []
    for n in range(10, 101, 10):
        graph = generate_graph(n, graph_type)
        graph_algo = GraphAlgorithms(graph)
        start_time = time.time()
        if algorithm == 'dijkstra':
            graph_algo.dijkstra(0)
        else:
            getattr(graph_algo, algorithm)()
        end_time = time.time()
        times.append(end_time - start_time)
    return times

def plot_execution_times(times, algorithm_name, graph_type):
    plt.plot(range(10, 101, 10), times, label=f"{algorithm_name} ({graph_type.capitalize()})")
    plt.legend()
    plt.xlabel("Number of Nodes")
    plt.ylabel("Execution Time (Seconds)")
    plt.title(f"Execution Time of {algorithm_name} on {graph_type.capitalize()} Graph")
    plt.show()

if __name__ == "__main__":
    algorithms = ['dijkstra', 'floyd_warshall']
    graph_types = ['sparse', 'dense']

    for graph_type in graph_types:
        plt.figure(figsize=(10, 6))
        for algorithm in algorithms:
            times = measure_algorithm_times(algorithm, graph_type)
            plt.plot(range(10, 101, 10), times, label=f"{algorithm.replace('_', ' ').title()} ({graph_type.capitalize()})")

        plt.legend()
        plt.xlabel("Number of Nodes")
        plt.ylabel("Execution Time (Seconds)")
        plt.title(f"Execution Time of Shortest Path Algorithms on {graph_type.capitalize()} Graph")
        plt.show()

    # Draw sample graphs
    for graph_type in graph_types:
        graph = generate_graph(10, graph_type)
        GraphAlgorithms(graph).draw_graph()
