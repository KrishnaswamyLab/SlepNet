import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from tqdm import tqdm

def create_clustered_dense_subgraph_graph(
    num_nodes=150,
    subgraph_size=50,
    signal_length=10,
    p_global=0.03,
    p_subgraph=0.6,
    label=0,
    informative_nodes=None  # ← now optionally passed
):
    """
    Creates a sparse ER graph with a densely connected informative subgraph embedded inside.
    """
    if informative_nodes is None:
        informative_nodes = list(range(subgraph_size))  # fixed by default

    G_global = nx.erdos_renyi_graph(num_nodes, p_global)
    while not nx.is_connected(G_global):
        G_global = nx.erdos_renyi_graph(num_nodes, p_global)

    # Dense subgraph generation
    G_dense = nx.erdos_renyi_graph(len(informative_nodes), p_subgraph)
    dense_edges = [
        (informative_nodes[u], informative_nodes[v])
        for u, v in G_dense.edges
    ]

    # Remove existing edges among informative nodes
    for i in range(len(informative_nodes)):
        for j in range(i + 1, len(informative_nodes)):
            if G_global.has_edge(informative_nodes[i], informative_nodes[j]):
                G_global.remove_edge(informative_nodes[i], informative_nodes[j])

    # Add dense subgraph edges
    G_global.add_edges_from(dense_edges)

    # Node features
    x = np.random.randn(num_nodes, signal_length) * 0.1
    signal = np.random.normal(loc=(1.0 if label else -1.0), scale=0.05, size=signal_length)
    for idx in informative_nodes:
        x[idx] += signal

    data = from_networkx(G_global)
    data.x = torch.tensor(x, dtype=torch.float)
    data.y = torch.tensor([label], dtype=torch.long)
    data.num_nodes = num_nodes
    data.informative_nodes = torch.tensor(informative_nodes, dtype=torch.long)

    return data

def generate_clustered_dense_subgraph_dataset(
    num_graphs=1000,
    informative_nodes=None,
    **kwargs
):
    dataset = []
    for i in tqdm(range(num_graphs), desc="Generating graphs with fixed informative subgraph"):
        label = 0 if i < num_graphs // 2 else 1
        g = create_clustered_dense_subgraph_graph(
            label=label,
            informative_nodes=informative_nodes,
            **kwargs
        )
        dataset.append(g)
    return dataset

# Example usage
if __name__ == "__main__":
    fixed_informative_nodes = list(range(50))  # nodes 0–29 will always be informative
    graphs = generate_clustered_dense_subgraph_dataset(
        num_graphs=1000,
        num_nodes=150,
        subgraph_size=50,
        signal_length=10,
        p_global=0.03,
        p_subgraph=0.6,
        informative_nodes=fixed_informative_nodes
    )
    torch.save(graphs, "toy_dataset_clustered_dense_fixed.pt")
    print("Saved 1000 graphs with fixed informative subgraph to toy_dataset_clustered_dense_fixed.pt")

