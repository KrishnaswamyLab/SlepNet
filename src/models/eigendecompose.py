import torch
import torch.nn as nn
import numpy as np
from torch_geometric.utils import to_dense_adj, get_laplacian
from scipy.sparse.linalg import eigsh
import random
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
# --------------------------
# Neural Eigenmapping Module
# --------------------------
class NeuralEigenmap(nn.Module):
    def __init__(self, num_nodes, hidden_dim=64, num_outputs=10):
        super(NeuralEigenmap, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_nodes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_outputs)
        )

    def forward(self, x):
        return self.net(x)

# --------------------------
# Compute Laplacian Eigenvectors
# --------------------------
def compute_laplacian_eigenvectors(data, num_vecs=10):
    from torch_geometric.utils import get_laplacian, to_dense_adj
    edge_index, edge_weight = get_laplacian(data.edge_index, normalization='sym')
    A = to_dense_adj(edge_index, edge_attr=edge_weight, max_num_nodes=data.num_nodes)[0].numpy()
    D = np.diag(A.sum(axis=1))
    L = D - A
    eigvals, eigvecs = eigsh(L, k=num_vecs, which='SM')
    return torch.tensor(eigvecs, dtype=torch.float)


def test_neural_eigenmap(model, eigvecs, test_idx):
    """
    Evaluates the trained NeuralEigenmap model on unseen (test) nodes.

    Args:
        model: trained NeuralEigenmap
        eigvecs: torch.Tensor of shape [N, K], ground truth Laplacian eigenvectors
        test_idx: list of indices for test nodes

    Returns:
        test_mse: Mean squared error on test nodes
        pred: predicted eigenvectors at test nodes
        target: ground truth eigenvectors at test nodes
    """
    model.eval()
    N = eigvecs.shape[0]
    identity_inputs = torch.eye(N)
    test_inputs = identity_inputs[test_idx]
    test_targets = eigvecs[test_idx]

    with torch.no_grad():
        pred = model(test_inputs)
        mse = F.mse_loss(pred, test_targets).item()

    print(f"Test MSE on {len(test_idx)} unseen nodes: {mse:.6f}")
    return mse, pred, test_targets

# --------------------------
# Training + Evaluation
# --------------------------
def train_neural_eigenmap_on_subset(data, num_vecs=10, train_ratio=0.5, epochs=300, lr=1e-3):
    N = data.num_nodes
    eigvecs = compute_laplacian_eigenvectors(data, num_vecs)

    # Create 1-hot inputs
    identity_inputs = torch.eye(N)

    # Split train/test
    indices = list(range(N))
    random.shuffle(indices)
    split = int(train_ratio * N)
    train_idx = indices[:split]
    test_idx = indices[split:]

    train_inputs = identity_inputs[train_idx]
    train_targets = eigvecs[train_idx]

    test_inputs = identity_inputs[test_idx]
    test_targets = eigvecs[test_idx]

    model = NeuralEigenmap(num_nodes=N, num_outputs=num_vecs)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        preds = model(train_inputs)
        loss = loss_fn(preds, train_targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                test_preds = model(test_inputs)
                test_loss = loss_fn(test_preds, test_targets).item()
            print(f"Epoch {epoch:03d} | Train Loss: {loss.item():.6f} | Test Loss: {test_loss:.6f}")

    return model, train_idx, test_idx, eigvecs

# --------------------------
# Run It
# --------------------------
if __name__ == "__main__":
    import torch
    import networkx as nx
    from torch_geometric.data import Data
    sizes = [500, 1000, 2000, 5000, 10000, 15000, 20000]
    time_eigh_list = []
    time_nn_list = []

    for N in sizes:
        print(f"\nProcessing graph with {N} nodes...")
        # Generate a random graph (Erdos-Renyi)
        G = nx.erdos_renyi_graph(N, p=min(0.1, 10.0/N))
        adj = nx.to_scipy_sparse_array(G).tocoo()
        # Convert to torch_geometric Data object

        row = torch.tensor(adj.row, dtype=torch.long)
        col = torch.tensor(adj.col, dtype=torch.long)
        edge_index = torch.stack([row, col], dim=0)
        edge_weight = torch.tensor(adj.data, dtype=torch.float)
        data = Data(edge_index=edge_index, edge_attr=edge_weight, num_nodes=N)

        num_vecs = min(10, N-2)
        model, train_idx, test_idx, eigvecs = train_neural_eigenmap_on_subset(data, num_vecs=num_vecs, epochs=100, lr=1e-3)
        
        def benchmark_eigh_vs_nn_inference(data, model, num_vecs=10):
            N = data.num_nodes
            identity_inputs = torch.eye(N)

            # Time eigsh
            start_eigh = time.time()
            eigvecs = compute_laplacian_eigenvectors(data, num_vecs)
            time_eigh = time.time() - start_eigh

            # Time neural net inference (forward pass only)
            model.eval()
            with torch.no_grad():
                start_nn = time.time()
                preds = model(identity_inputs)
                time_nn = time.time() - start_nn

            return time_eigh, time_nn

        t_eigh, t_nn = benchmark_eigh_vs_nn_inference(data, model, num_vecs=num_vecs)
        print(f"eigsh runtime: {t_eigh:.6f} sec | NeuralEigenmap inference runtime: {t_nn:.6f} sec")
        time_eigh_list.append(t_eigh)
        time_nn_list.append(t_nn)

    # Plot runtime comparison (log scale for y)
    plt.figure(figsize=(7, 5))
    plt.plot(sizes, time_eigh_list, 'o-r', label='eigsh')
    plt.plot(sizes, time_nn_list, 'o-b', label='NeuralEigenmap')
    plt.xlabel("Number of nodes")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime vs Number of Nodes")
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig("runtime_vs_nodes_updated.png")
    plt.show()
