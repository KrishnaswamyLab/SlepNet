import torch
import torch.nn as nn
import torch.nn.functional as F
from src.spectral_layer import SpectralConv_Batch
from torch_geometric.nn import global_mean_pool

class SpectralNet(nn.Module):
    def __init__(
        self,
        num_nodes,
        num_features,
        num_classes,
        hidden_dim=16,
        num_eigenvectors=10,
        adj=None
    ):
        super(SpectralNet, self).__init__()

        self.spectral_conv = SpectralConv_Batch(
            in_dim=num_features,
            out_dim=hidden_dim,
            num_eigenvectors=num_eigenvectors,
            hidden_dim=hidden_dim,
            adj=adj
        )

        self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, H, edge_index, batch):
        H = self.spectral_conv(H, edge_index, batch)
        H = global_mean_pool(H, batch)
        logits = self.linear(H)
        return logits
