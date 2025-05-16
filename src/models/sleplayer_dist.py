import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data
import networkx as nx
import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from torch_geometric.utils import to_dense_adj


class Sleplayer_Distance(nn.Module):
    def __init__(self, num_nodes, num_clusters, cluster_masks, num_slepians, in_channels, hidden_dim, out_channels):
        super(Sleplayer_Distance, self).__init__()
        self.num_nodes = num_nodes
        self.num_slepians = num_slepians
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels

        self.W1 = nn.Parameter(torch.randn(num_slepians, in_channels, hidden_dim))
        self.W2 = nn.Parameter(torch.randn(num_slepians, hidden_dim, hidden_dim))
        self.W3 = nn.Parameter(torch.randn(num_slepians, hidden_dim, out_channels))

        self.cluster_attention = nn.Parameter(torch.randn(num_clusters, 1), requires_grad=True)
        self.register_buffer("cluster_masks", cluster_masks)

    def compute_slepian_basis(self, adj, mask):
        D = torch.diag(torch.sum(adj, dim=1))
        L = D - adj + 1e-5 * torch.eye(self.num_nodes, device=adj.device)

        eigvals, U = torch.linalg.eigh(L)

        threshold = mask.mean().item()
        binary_mask = (mask > threshold).float()
        S = torch.diag(binary_mask)

        W = torch.eye(self.num_nodes, device=adj.device)[:, :self.num_slepians]
        Lambda = torch.diag(eigvals)
        Lambda_W = W.T @ Lambda @ W
        Lambda_W_sqrt = torch.linalg.matrix_power(Lambda_W, 1) ** 0.5

        temp = U @ W
        C = Lambda_W_sqrt @ (temp.T @ S @ temp) @ Lambda_W_sqrt

        _, s = torch.linalg.eigh(C)
        s_k = s[:, :self.num_slepians]
        s_k = U @ W @ s_k
        return s_k

    def laplacian_smoothness(self, adj, mask):
        D = torch.diag(torch.sum(adj, dim=1))
        L = D - adj
        m = torch.sigmoid(mask).view(-1, 1)
        return (m.T @ L @ m).squeeze()

    def forward(self, H, edge_index):
        adj = to_dense_adj(edge_index)[0]
        weights = torch.sigmoid(self.cluster_attention)  # [num_clusters, 1]
        combined_mask = (weights.T @ self.cluster_masks).squeeze()  # [num_nodes]

        s_k = self.compute_slepian_basis(adj, combined_mask)  # [num_nodes, num_slepians]

        H_slep = s_k.T @ H  # [num_slepians, in_channels] == (k, in)

        H_hidden1 = torch.einsum("kic,ki->kc", self.W1, H_slep)  # (k, hidden)
        H_hidden1 = F.relu(H_hidden1)

        H_hidden2 = torch.einsum("kch,kc->kc", self.W2, H_hidden1)  # (k, hidden)
        H_hidden2 = F.relu(H_hidden2)

        H_filtered = torch.einsum("kho,kh->ko", self.W3, H_hidden2)  # (k, out_channels)

        H_out = s_k @ H_filtered  # [num_nodes, out_channels]
        return H_out

        