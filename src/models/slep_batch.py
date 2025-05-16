import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj

class Sleplayer_Energy_Batch(nn.Module):
    def __init__(self, num_nodes, num_clusters, cluster_masks, num_slepians,
                 in_channels, hidden_dim, out_channels, adj):
        super(Sleplayer_Energy_Batch, self).__init__()
        self.num_nodes = num_nodes
        self.num_slepians = num_slepians
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels

        # Spectral filters: 3-layer MLP as learnable parameters per slepian
        self.W1 = nn.Parameter(torch.randn(num_slepians, in_channels, hidden_dim))
        self.W2 = nn.Parameter(torch.randn(num_slepians, hidden_dim, hidden_dim))
        self.W3 = nn.Parameter(torch.randn(num_slepians, hidden_dim, out_channels))

        # Cluster attention for soft subgraph selection
        self.cluster_attention = nn.Parameter(torch.randn(num_clusters, 1), requires_grad=True)
        self.register_buffer("cluster_masks", cluster_masks)  # [num_clusters, num_nodes]

        # Precompute Laplacian and eigenvectors since adj is shared
        D = torch.diag(torch.sum(adj, dim=1))
        L = D - adj + 1e-5 * torch.eye(num_nodes, device=adj.device)
        eigvals, U = torch.linalg.eigh(L)  # [N, N]

        self.register_buffer("adj", adj)
        self.register_buffer("U", U)  # Laplacian eigenvectors
        self.register_buffer("W", torch.eye(num_nodes, device=adj.device)[:, :num_slepians])  # bandlimiting

    def forward(self, H, edge_index, batch):
        # H: [total_nodes, in_channels], batch: [total_nodes]
        # Reshape into [batch_size, num_nodes, in_channels]
        batch_size = batch.max().item() + 1
        H = H.view(batch_size, self.num_nodes, self.in_channels)

        # Learn spatial subgraph mask
        weights = torch.sigmoid(self.cluster_attention)  # [num_clusters, 1]
        combined_mask = (weights.T @ self.cluster_masks).squeeze()  # [num_nodes]
        # threshold = combined_mask.mean().item()
        threshold = 0.5
        # import pdb; pdb.set_trace()
        binary_mask = (combined_mask > threshold).float()
        S = torch.diag(binary_mask)  # [N, N]

        # Compute Slepian basis using shared Laplacian
        C = self.W.T @ self.U.T @ S @ self.U @ self.W  # [k, k]
        _, s = torch.linalg.eigh(C)
        s_k = s[:, :self.num_slepians]  # [k, k]
        s_k = self.U @ self.W @ s_k  # [N, k]
        # import pdb; pdb.set_trace()
        # Spectral transform of signals
        H_slep = torch.einsum("bnf,nk->bkf", H, s_k)

        # Apply MLP in spectral domain
        H1 = torch.einsum("kic,bki->bkc", self.W1, H_slep)
        H1 = F.relu(H1)
        H2 = torch.einsum("kch,bkc->bkh", self.W2, H1)
        H2 = F.relu(H2)
        H_filtered = torch.einsum("kho,bkh->bko", self.W3, H2)  # [B, k, out_channels]

        # Transform back to spatial domain
        H_out = torch.einsum("nk,bko->bno", s_k, H_filtered)  # [B, N, out_channels]

        # Flatten to [B*N, out_channels] to match PyG expectations
        return H_out.reshape(-1, self.out_channels)


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj

class Sleplayer_Distance_Batch(nn.Module):
    def __init__(self, num_nodes, num_clusters, cluster_masks, num_slepians,
                 in_channels, hidden_dim, out_channels, adj):
        super(Sleplayer_Distance_Batch, self).__init__()
        self.num_nodes = num_nodes
        self.num_slepians = num_slepians
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels

        # Spectral filters: 3-layer MLP as learnable parameters per slepian
        self.W1 = nn.Parameter(torch.randn(num_slepians, in_channels, hidden_dim))
        self.W2 = nn.Parameter(torch.randn(num_slepians, hidden_dim, hidden_dim))
        self.W3 = nn.Parameter(torch.randn(num_slepians, hidden_dim, out_channels))

        # Cluster attention for soft subgraph selection
        self.cluster_attention = nn.Parameter(torch.randn(num_clusters, 1), requires_grad=True)
        self.register_buffer("cluster_masks", cluster_masks)

        # Precompute Laplacian and eigenvectors since adj is shared
        D = torch.diag(torch.sum(adj, dim=1))
        L = D - adj + 1e-5 * torch.eye(num_nodes, device=adj.device)
        eigvals, U = torch.linalg.eigh(L)  # [N, N]

        self.register_buffer("adj", adj)
        self.register_buffer("U", U)
        self.register_buffer("Lambda", torch.diag(eigvals))
        self.register_buffer("W", torch.eye(num_nodes, device=adj.device)[:, :num_slepians])

    def forward(self, H, edge_index, batch):
        batch_size = batch.max().item() + 1
        H = H.view(batch_size, self.num_nodes, self.in_channels)  # [B, N, F]

        # Learn spatial subgraph mask
        weights = torch.sigmoid(self.cluster_attention)  # [num_clusters, 1]
        combined_mask = (weights.T @ self.cluster_masks).squeeze()  # [N]
        # threshold = combined_mask.mean().item()
        threshold = 0.5
        binary_mask = (combined_mask > threshold).float()
        S = torch.diag(binary_mask)  # [N, N]

        # Compute Slepian basis using distance-aware formulation
        Lambda_W = self.W.T @ self.Lambda @ self.W  # [k, k]
        Lambda_W_sqrt = Lambda_W.sqrt()  # [k, k]
        temp = self.U @ self.W  # [N, k]
        C = Lambda_W_sqrt @ (temp.T @ S @ temp) @ Lambda_W_sqrt  # [k, k]

        _, s = torch.linalg.eigh(C)
        s_k = s[:, :self.num_slepians]  # [k, k]
        s_k = self.U @ self.W @ s_k  # [N, k]

        # Spectral transform of signals
        H_slep = torch.einsum("bnf,nk->bkf", H, s_k)  # [B, k, F]

        # Apply MLP in spectral domain
        H1 = torch.einsum("kic,bki->bkc", self.W1, H_slep)
        H1 = F.relu(H1)
        H2 = torch.einsum("kch,bkc->bkh", self.W2, H1)
        H2 = F.relu(H2)
        H_filtered = torch.einsum("kho,bkh->bko", self.W3, H2)  # [B, k, out_channels]

        # Transform back to spatial domain
        H_out = torch.einsum("nk,bko->bno", s_k, H_filtered)  # [B, N, out_channels]

        return H_out.reshape(-1, self.out_channels)

