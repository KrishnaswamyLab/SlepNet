import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj

class SpectralConv_Batch(nn.Module):
    def __init__(self, in_dim, out_dim, num_eigenvectors=10, hidden_dim=32, adj=None):
        super(SpectralConv_Batch, self).__init__()
        self.num_eigenvectors = num_eigenvectors

        # Spectral filters: 3-layer MLP as learnable parameters per eigenvector
        self.W1 = nn.Parameter(torch.randn(num_eigenvectors, in_dim, hidden_dim))
        self.W2 = nn.Parameter(torch.randn(num_eigenvectors, hidden_dim, hidden_dim))
        self.W3 = nn.Parameter(torch.randn(num_eigenvectors, hidden_dim, out_dim))

        # Precompute Laplacian eigenvectors
        if adj is not None:
            adj = adj.float()
            degree = torch.diag(adj.sum(dim=1))
            laplacian_dense = degree - adj + 1e-5 * torch.eye(adj.size(0), device=adj.device)
            eigenvalues, eigenvectors = torch.linalg.eigh(laplacian_dense)

            self.register_buffer("eigenvectors", eigenvectors[:, :num_eigenvectors])  # [N, k]
        else:
            raise ValueError("Adjacency matrix must be provided for shared Laplacian eigenvectors.")

    def forward(self, x, edge_index, batch):
        batch_size = batch.max().item() + 1
        num_nodes = self.eigenvectors.size(0)
        # import pdb;pdb.set_trace()
        in_dim = x.size(1)

        # Reshape x to [batch_size, num_nodes, in_dim]
        x = x.view(batch_size, num_nodes, in_dim)

        # Project node features into the spectral domain
        x_spec = torch.einsum("bnf,nk->bkf", x, self.eigenvectors)  # [batch, k, in_dim]

        # Apply 3-layer MLP in spectral domain
        H1 = torch.einsum("kic,bki->bkc", self.W1, x_spec)
        H1 = F.relu(H1)
        H2 = torch.einsum("kch,bkc->bkh", self.W2, H1)
        H2 = F.relu(H2)
        x_filtered = torch.einsum("kho,bkh->bko", self.W3, H2)  # [batch, k, out_dim]

        # Inverse Graph Fourier Transform
        x_out = torch.einsum("nk,bko->bno", self.eigenvectors, x_filtered)  # [batch, num_nodes, out_dim]

        return x_out.reshape(-1, x_out.size(-1))
