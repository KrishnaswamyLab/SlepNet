import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from tqdm import tqdm

def create_three_er_graph(num_nodes_per_group=50, signal_length=10, p=0.3, label=0):
    total_nodes = 3 * num_nodes_per_group

    # Create 3 disconnected ER graphs
    G1 = nx.erdos_renyi_graph(n=num_nodes_per_group, p=p)
    G2 = nx.erdos_renyi_graph(n=num_nodes_per_group, p=p)
    G3 = nx.erdos_renyi_graph(n=num_nodes_per_group, p=p)

    # Relabel nodes to be in disjoint sets: 0–49, 50–99, 100–149
    G2 = nx.relabel_nodes(G2, lambda x: x + num_nodes_per_group)
    G3 = nx.relabel_nodes(G3, lambda x: x + 2 * num_nodes_per_group)

    # Combine all
    G = nx.Graph()
    G.add_nodes_from(G1.nodes)
    G.add_nodes_from(G2.nodes)
    G.add_nodes_from(G3.nodes)
    G.add_edges_from(G1.edges)
    G.add_edges_from(G2.edges)
    G.add_edges_from(G3.edges)

    # Add inter-group connections (random 4 edges between groups)
    rng = np.random.default_rng()
    for _ in range(4):
        G.add_edge(
            rng.integers(0, num_nodes_per_group),
            rng.integers(num_nodes_per_group, 2 * num_nodes_per_group)
        )
        G.add_edge(
            rng.integers(num_nodes_per_group, 2 * num_nodes_per_group),
            rng.integers(2 * num_nodes_per_group, 3 * num_nodes_per_group)
        )

    # Node features
    x = np.random.randn(total_nodes, signal_length) * 0.1  # Base noise for all

    if label == 0:
        informative_signal = np.random.normal(loc=-1.0, scale=0.1, size=signal_length)
    else:
        informative_signal = np.random.normal(loc=1.0, scale=0.1, size=signal_length)

    # Add informative signal to Group 2 (nodes 50–99)
    # x[num_nodes_per_group:2 * num_nodes_per_group] += informative_signal

    # Add informative signal to Group 1 (nodes 0–49)
    x[0:num_nodes_per_group] += informative_signal

    # Convert to PyG Data object
    data = from_networkx(G)
    data.x = torch.tensor(x, dtype=torch.float)
    data.y = torch.tensor([label], dtype=torch.long)
    data.num_nodes = total_nodes

    return data

def generate_three_er_dataset(num_graphs=1000, num_nodes_per_group=50, signal_length=10, p=0.3):
    dataset = []
    for i in tqdm(range(num_graphs), desc="Generating 3-ER graphs"):
        label = 0 if i < num_graphs // 2 else 1
        graph = create_three_er_graph(num_nodes_per_group, signal_length, p, label)
        dataset.append(graph)
    return dataset

# Usage
if __name__ == "__main__":
    graphs = generate_three_er_dataset(num_graphs=1000, num_nodes_per_group=50, signal_length=10, p=0.3)
    torch.save(graphs, "toy_dataset_3ER_1000_new.pt")
    print("Saved 1000 ER-component graphs to toy_dataset_3ER_1000.pt")
