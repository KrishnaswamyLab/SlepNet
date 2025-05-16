import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm
# Load node signals and adjacency matrix
signals = np.load('PEMS08/graph_signals.npy')  # shape: [T, N]
adj = np.load('PEMS08/adjacency_matrix.npy')   # shape: [N, N]
labels = np.load('PEMS08/DAY/label.npy')        # shape: [T, N]

def adj_to_edge_index(adj):
    row, col = np.nonzero(adj)
    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_weight = torch.tensor(adj[row, col], dtype=torch.float)
    return edge_index, edge_weight

def create_graphs(signals, labels, edge_index, edge_weight=None):
    graphs = []
    # import pdb; pdb.set_trace()
    for t in tqdm(range(signals.shape[0])):
        x = torch.tensor(signals[t].squeeze(-1), dtype=torch.float).unsqueeze(1)
        y = torch.tensor(labels[t], dtype=torch.float)   # [N, 1] or [N]
        data = Data(x=x, edge_index=edge_index, y=y)
        if edge_weight is not None:
            data.edge_weight = edge_weight
        graphs.append(data)
    return graphs

# Prepare edge_index and edge_weight
edge_index, edge_weight = adj_to_edge_index(adj)

# Create PyG graphs
graphs = create_graphs(signals, labels, edge_index, edge_weight)
# import pdb; pdb.set_trace()
# (Optional) Save the graph list for fast loading later
torch.save(graphs, 'PEMS08/graphs.pt')
