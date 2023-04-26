import time
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def dijkstra(graph, start):
    dist = {node: float('inf') for node in graph}
    dist[start] = 0
    visited = set()

    while len(visited) != len(graph):
        min_node = None
        for node in dist:
            if node not in visited:
                if min_node is None or dist[node] < dist[min_node]:
                    min_node = node
        visited.add(min_node)

        for neighbor, weight in graph[min_node].items():
            new_dist = dist[min_node] + weight
            if new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
    return dist

def floyd(graph):
    nodes = list(graph.keys())
    n = len(nodes)
    dist = np.full((n, n), np.inf)

    for i, node in enumerate(nodes):
        dist[i, i] = 0
        for neighbor, weight in graph[node].items():
            j = nodes.index(neighbor)
            dist[i, j] = weight

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i, k] + dist[k, j] < dist[i, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]

    return {nodes[i]: {nodes[j]: dist[i, j] for j in range(n)} for i in range(n)}

def generate_random_graph(num_nodes, max_weight=10):
    graph = {i: {} for i in range(num_nodes)}
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            weight = random.randint(1, max_weight)
            graph[i][j] = weight
            graph[j][i] = weight
    return graph

def plot_comparison(nodes, dijkstra_times, floyd_times):
    plt.plot(nodes, dijkstra_times, label="Dijkstra")
    plt.plot(nodes, floyd_times, label="Floyd")
    plt.xlabel("Nr of Nodes")
    plt.ylabel("Execution Time (s)")
    plt.legend()
    plt.title("Dijkstra vs Floyd Execution Time Comparison")
    plt.show()

def draw_graph_tree(graph, title):
    G = nx.Graph(graph)
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', font_size=10, font_weight='bold', node_size=700)
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(i, j): graph[i][j] for i in graph for j in graph[i]})
    plt.title(title)
    plt.show()

def main():
    num_nodes = [5, 10, 15, 20, 25]
    dijkstra_times = []
    floyd_times = []

    for n in num_nodes:
        graph = generate_random_graph(n)
        start_time = time.time()
        dijkstra(graph, 0)
        dijkstra_times.append(time.time() - start_time)

        start_time = time.time()
        floyd(graph)
        floyd_times.append(time.time() - start_time)

    plot_comparison(num_nodes, dijkstra_times, floyd_times)

    sample_graph = generate_random_graph(10)
    draw_graph_tree(sample_graph, "Sample Graph for Dijkstra and Floyd Algorithms")

    dijkstra_tree = dijkstra(sample_graph, 0)
    draw_graph_tree({node: {neighbor: sample_graph[node][neighbor] for neighbor in sample_graph[node] if
                            dijkstra_tree[neighbor] == dijkstra_tree[node] + sample_graph[node][neighbor]} for node in
                     sample_graph}, "Dijkstra's Algorithm Graph Tree")

    floyd_tree = floyd(sample_graph)
    draw_graph_tree({node: {neighbor: sample_graph[node][neighbor] for neighbor in sample_graph[node] if
                            floyd_tree[node][neighbor] == sample_graph[node][neighbor]} for node in sample_graph},
                    "Floyd's Algorithm Graph Tree")


if __name__ == "__main__":
    main()
