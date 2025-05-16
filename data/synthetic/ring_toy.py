import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from tqdm import tqdm

def create_three_ring_graph(num_nodes_per_ring=50, signal_length=10, label=0):
    total_nodes = 3 * num_nodes_per_ring
    adj = np.zeros((total_nodes, total_nodes))

    # Inner ring: 0–49
    for i in range(num_nodes_per_ring):
        j = (i + 1) % num_nodes_per_ring
        adj[i, j] = 1
        adj[j, i] = 1

    # Middle ring: 50–99
    for i in range(num_nodes_per_ring, 2 * num_nodes_per_ring):
        j = (i + 1 - num_nodes_per_ring) % num_nodes_per_ring + num_nodes_per_ring
        adj[i, j] = 1
        adj[j, i] = 1

    # Outer ring: 100–149
    for i in range(2 * num_nodes_per_ring, total_nodes):
        j = (i + 1 - 2 * num_nodes_per_ring) % num_nodes_per_ring + 2 * num_nodes_per_ring
        adj[i, j] = 1
        adj[j, i] = 1

    # Radial connections
    for i in range(num_nodes_per_ring):
        # Inner ↔ Middle
        adj[i, i + num_nodes_per_ring] = 1
        adj[i + num_nodes_per_ring, i] = 1
        # Middle ↔ Outer
        adj[i + num_nodes_per_ring, i + 2 * num_nodes_per_ring] = 1
        adj[i + 2 * num_nodes_per_ring, i + num_nodes_per_ring] = 1

    edge_index, _ = dense_to_sparse(torch.tensor(adj, dtype=torch.float))

    # Node features
    if label == 0:
        base_signal = np.random.normal(loc=0.0, scale=1.0, size=signal_length)
    else:
        base_signal = np.random.normal(loc=1.0, scale=0.5, size=signal_length)
    # noisy_signal = base_signal + np.random.randn(signal_length) * 0.05  # add slight noise

    x = np.random.randn(total_nodes, signal_length) * 0.1  # base noise for all

    # Middle ring gets informative signal
    x[num_nodes_per_ring:2*num_nodes_per_ring] += base_signal

    return Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=edge_index,
        y=torch.tensor([label], dtype=torch.long),
        num_nodes=total_nodes
    )

def generate_three_ring_dataset(num_graphs=1000, num_nodes_per_ring=50, signal_length=10):
    dataset = []
    for i in tqdm(range(num_graphs), desc="Generating 3-ring graphs"):
        label = 0 if i < num_graphs // 2 else 1
        graph = create_three_ring_graph(num_nodes_per_ring, signal_length, label)
        dataset.append(graph)
    return dataset

# Example usage
if __name__ == "__main__":
    graphs = generate_three_ring_dataset(num_graphs=1000, num_nodes_per_ring=50, signal_length=10)
    torch.save(graphs, "toy_dataset_3ring_1000.pt")
    print("Saved 1000 3-ring graphs to toy_dataset_3ring_1000.pt")
