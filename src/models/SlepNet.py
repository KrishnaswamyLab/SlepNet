import torch
import torch.nn as nn
import torch.nn.functional as F
from src.sleplayer import Sleplayer_Energy
from src.sleplayer_dist import Sleplayer_Distance
from src.slep_batch import Sleplayer_Energy_Batch, Sleplayer_Distance_Batch
from src.slep_batch_mask import Sleplayer_Distance_Batch_New, Sleplayer_Energy_Batch_New
from torch_geometric.nn import global_mean_pool


class SlepNet(nn.Module):
    def __init__(
        self,
        num_nodes,
        num_features,
        num_clusters,
        cluster_masks,
        num_slepians,
        num_classes,
        hidden_dim=16,
        layer_type="distance",  # 'emb' or 'base'
        adj = None
    ):
        super(SlepNet, self).__init__()

        if layer_type == "distance":
            LayerClass = Sleplayer_Distance
            self.sleplayer = LayerClass(
            num_nodes=num_nodes,
            num_clusters=num_clusters,
            cluster_masks=cluster_masks,
            num_slepians=num_slepians,
            in_channels=num_features,
            hidden_dim=hidden_dim,
            out_channels=hidden_dim,
        )

        elif layer_type == "energy":
            LayerClass = Sleplayer_Energy
            self.sleplayer = LayerClass(
            num_nodes=num_nodes,
            num_clusters=num_clusters,
            cluster_masks=cluster_masks,
            num_slepians=num_slepians,
            in_channels=num_features,
            hidden_dim=hidden_dim,
            out_channels=hidden_dim,
            )
        elif layer_type == "batch_energy":
            LayerClass = Sleplayer_Energy_Batch
            self.sleplayer = LayerClass(
            num_nodes=num_nodes,
            num_clusters=num_clusters,
            cluster_masks=cluster_masks,
            num_slepians=num_slepians,
            in_channels=num_features,
            hidden_dim=hidden_dim,
            out_channels=hidden_dim,
            adj = adj
            )
        elif layer_type == "batch_distance":
            LayerClass = Sleplayer_Distance_Batch
            self.sleplayer = LayerClass(
            num_nodes=num_nodes,
            num_clusters=num_clusters,
            cluster_masks=cluster_masks,
            num_slepians=num_slepians,
            in_channels=num_features,
            hidden_dim=hidden_dim,
            out_channels=hidden_dim,
            adj = adj
        )
        
        elif layer_type == "attn_energy":
            LayerClass = Sleplayer_Energy_Batch_New
            self.sleplayer = LayerClass(
            num_nodes=num_nodes,
            num_clusters=num_clusters,
            cluster_masks=cluster_masks,
            num_slepians=num_slepians,
            in_channels=num_features,
            hidden_dim=hidden_dim,
            out_channels=hidden_dim,
            adj = adj
        )
        elif layer_type == "attn_distance":
            LayerClass = Sleplayer_Distance_Batch_New
            self.sleplayer = LayerClass(
            num_nodes=num_nodes,
            num_clusters=num_clusters,
            cluster_masks=cluster_masks,
            num_slepians=num_slepians,
            in_channels=num_features,
            hidden_dim=hidden_dim,
            out_channels=hidden_dim,
            adj = adj
        )

        else:
            raise ValueError(f"Unknown layer_type: {layer_type}. Choose from ['distance', 'energy'].")

        
        self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, H, edge_index, batch):
        H = self.sleplayer(H, edge_index, batch)
        H = global_mean_pool(H, batch)
        logits = self.linear(H)
        return logits
