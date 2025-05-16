import torch
import torch.nn as nn
import torch.nn.functional as F

class Sleplayer_Energy_Batch_New(nn.Module):
    def __init__(self, num_nodes, num_clusters, cluster_masks, num_slepians,
                 in_channels, hidden_dim, out_channels, adj):
        super(Sleplayer_Energy_Batch_New, self).__init__()
        self.num_nodes = num_nodes
        self.num_clusters = num_clusters
        self.num_slepians = num_slepians
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels

        # Spectral filters
        self.W1 = nn.Parameter(torch.randn(num_slepians, in_channels, hidden_dim))
        self.W2 = nn.Parameter(torch.randn(num_slepians, hidden_dim, hidden_dim))
        self.W3 = nn.Parameter(torch.randn(num_slepians, hidden_dim, out_channels))

        # MLP for attention over cluster summaries
        self.att_mlp = nn.Sequential(
            nn.Linear(in_channels, 1),
            nn.Sigmoid()
        )

        self.register_buffer("cluster_masks", cluster_masks)  # [num_clusters, num_nodes]

        # Precompute Laplacian and eigenvectors
        D = torch.diag(torch.sum(adj, dim=1))
        L = D - adj + 1e-5 * torch.eye(num_nodes, device=adj.device)
        eigvals, U = torch.linalg.eigh(L)

        self.register_buffer("adj", adj)
        self.register_buffer("U", U)
        self.register_buffer("W", torch.eye(num_nodes, device=adj.device)[:, :num_slepians])

    def forward(self, H, edge_index, batch):
        batch_size = batch.max().item() + 1
        H = H.view(batch_size, self.num_nodes, self.in_channels)  # [B, N, F]

        # Compute aggregated cluster-level features H_i = M_i^T X
        cluster_feats = torch.einsum("cn,bnf->bcf", self.cluster_masks, H)  # [B, C, F]

        # Compute attention per cluster
        att_scores = self.att_mlp(cluster_feats).squeeze(-1)  # [B, C]

        # Compute soft node mask: m = a^T M
        node_mask = torch.einsum("bc,cn->bn", att_scores, self.cluster_masks)  # [B, N]
        binary_mask = (node_mask > 0.5).float()

        S = torch.stack([torch.diag(mask) for mask in binary_mask])  # [B, N, N]

        # Shared Slepian basis from first sample (optional to vectorize)
        C = self.W.T @ self.U.T @ S[0] @ self.U @ self.W
        _, s = torch.linalg.eigh(C)
        s_k = s[:, :self.num_slepians]
        s_k = self.U @ self.W @ s_k  # [N, k]

        # Spectral transform of signals
        H_slep = torch.einsum("bnf,nk->bkf", H, s_k)

        # Apply MLP in spectral domain
        H1 = F.relu(torch.einsum("kic,bki->bkc", self.W1, H_slep))
        H2 = F.relu(torch.einsum("kch,bkc->bkh", self.W2, H1))
        H_filtered = torch.einsum("kho,bkh->bko", self.W3, H2)

        # Transform back to spatial domain
        H_out = torch.einsum("nk,bko->bno", s_k, H_filtered)
        return H_out.reshape(-1, self.out_channels)


import torch
import torch.nn as nn
import torch.nn.functional as F

class Sleplayer_Distance_Batch_New(nn.Module):
    def __init__(self, num_nodes, num_clusters, cluster_masks, num_slepians,
                 in_channels, hidden_dim, out_channels, adj):
        super(Sleplayer_Distance_Batch_New, self).__init__()
        self.num_nodes = num_nodes
        self.num_clusters = num_clusters
        self.num_slepians = num_slepians
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels

        # Spectral filters
        self.W1 = nn.Parameter(torch.randn(num_slepians, in_channels, hidden_dim))
        self.W2 = nn.Parameter(torch.randn(num_slepians, hidden_dim, hidden_dim))
        self.W3 = nn.Parameter(torch.randn(num_slepians, hidden_dim, out_channels))

        # MLP for attention over cluster summaries
        self.att_mlp = nn.Sequential(
            nn.Linear(in_channels, 1),
            nn.Sigmoid()
        )

        self.register_buffer("cluster_masks", cluster_masks)  # [num_clusters, num_nodes]

        # Precompute Laplacian and eigenvectors
        D = torch.diag(torch.sum(adj, dim=1))
        L = D - adj + 1e-5 * torch.eye(num_nodes, device=adj.device)
        eigvals, U = torch.linalg.eigh(L)

        self.register_buffer("adj", adj)
        self.register_buffer("U", U)
        self.register_buffer("W", torch.eye(num_nodes, device=adj.device)[:, :num_slepians])
        self.register_buffer("Lambda", torch.diag(eigvals))

    def forward(self, H, edge_index, batch):
        batch_size = batch.max().item() + 1
        H = H.view(batch_size, self.num_nodes, self.in_channels)  # [B, N, F]

        # Compute aggregated cluster-level features H_i = M_i^T X
        cluster_feats = torch.einsum("cn,bnf->bcf", self.cluster_masks, H)  # [B, C, F]

        # Compute attention per cluster
        att_scores = self.att_mlp(cluster_feats).squeeze(-1)  # [B, C]

        # Compute soft node mask: m = a^T M
        node_mask = torch.einsum("bc,cn->bn", att_scores, self.cluster_masks)  # [B, N]
        binary_mask = (node_mask > 0.5).float()

        S = torch.stack([torch.diag(mask) for mask in binary_mask])  # [B, N, N]

        # Shared Slepian basis from first sample
        Lambda_W = self.W.T @ self.Lambda @ self.W
        Lambda_W_sqrt = Lambda_W.sqrt()
        temp = self.U @ self.W
        C = Lambda_W_sqrt @ (temp.T @ S[0] @ temp) @ Lambda_W_sqrt

        _, s = torch.linalg.eigh(C)
        s_k = s[:, :self.num_slepians]
        s_k = self.U @ self.W @ s_k  # [N, k]

        # Spectral transform of signals
        H_slep = torch.einsum("bnf,nk->bkf", H, s_k)  # [B, k, F]

        # Apply MLP in spectral domain
        H1 = F.relu(torch.einsum("kic,bki->bkc", self.W1, H_slep))
        H2 = F.relu(torch.einsum("kch,bkc->bkh", self.W2, H1))
        H_filtered = torch.einsum("kho,bkh->bko", self.W3, H2)  # [B, k, out_channels]

        # Transform back to spatial domain
        H_out = torch.einsum("nk,bko->bno", s_k, H_filtered)  # [B, N, out_channels]

        return H_out.reshape(-1, self.out_channels)