import numpy as np
import torch
import matplotlib.pyplot as plt
from nilearn import plotting
from torch_geometric.utils import to_dense_adj
import pandas as pd
from src.SlepNet import SlepNet
from scipy.ndimage import gaussian_filter
import nibabel as nib
from nilearn import datasets
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from torch_geometric.utils import to_networkx

def draw_attention_highlight_ring_graph_full(
    data, attention_mask, title="Attention Over 3-Ring Graph", cmap='viridis'
):
    """
    Draws a 3-ring graph and highlights all nodes using attention weights (not just top-k).
    
    Args:
        data: PyG Data object with 150 nodes
        attention_mask: torch.Tensor or np.array of shape [150]
        title: Plot title
        cmap: Matplotlib colormap
        save_path: PNG output path
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    from torch_geometric.utils import to_networkx

    G = to_networkx(data, to_undirected=True)
    pos = {}
    num_nodes_per_ring = data.num_nodes // 3

    # Arrange nodes in concentric rings
    for i in range(num_nodes_per_ring):
        angle = 2 * np.pi * i / num_nodes_per_ring
        pos[i] = (np.cos(angle), np.sin(angle))                               # Inner ring
        pos[i + num_nodes_per_ring] = (1.8 * np.cos(angle), 1.8 * np.sin(angle))  # Middle
        pos[i + 2 * num_nodes_per_ring] = (2.6 * np.cos(angle), 2.6 * np.sin(angle))  # Outer

    # Normalize attention values
    attention_np = attention_mask.cpu().numpy() if isinstance(attention_mask, torch.Tensor) else attention_mask
    attention_norm = (attention_np - attention_np.min()) / (attention_np.max() - attention_np.min() + 1e-8)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_title(title)

    # Draw nodes with colormap
    nodes = nx.draw_networkx_nodes(
        G, pos=pos,
        node_color=attention_norm,
        cmap=plt.get_cmap(cmap),
        node_size=300,
        edgecolors='k',
        linewidths=0.2,
        ax=ax
    )

    nx.draw_networkx_edges(G, pos=pos, edge_color='lightgray', ax=ax)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Attention Weight", rotation=270, labelpad=15)
    cbar.ax.tick_params(labelsize=8)

    ax.axis('off')
    plt.tight_layout()
    plt.savefig("RING_GRAPH", dpi=300)
    plt.show()


import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from torch_geometric.utils import to_networkx

def draw_attention_highlight_er_graph_full(
    data, attention_mask, title="Attention Weights on ER Graph", cmap='viridis'
):
    """
    Draws an ER-style graph and highlights all nodes using a continuous attention colormap.
    Compatible with older networkx versions (no `norm`).
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    from torch_geometric.utils import to_networkx

    G = to_networkx(data, to_undirected=True)
    pos = nx.spring_layout(G, seed=42)

    attention_np = attention_mask.cpu().numpy() if isinstance(attention_mask, torch.Tensor) else attention_mask
    # Normalize to [0, 1]
    attention_norm = (attention_np - attention_np.min()) / (attention_np.max() - attention_np.min() + 1e-8)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(title)

    nodes = nx.draw_networkx_nodes(
        G, pos=pos,
        node_color=attention_norm,
        cmap=plt.get_cmap(cmap),
        node_size=300,
        edgecolors='k',
        linewidths=0.2,
        ax=ax
    )

    nx.draw_networkx_edges(G, pos=pos, edge_color='lightgray', ax=ax)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Attention Weight", rotation=270, labelpad=15)
    cbar.ax.tick_params(labelsize=8)

    ax.axis('off')
    plt.tight_layout()
    plt.savefig("ER_GRAPH", dpi=300)
    plt.show()



def draw_attention_mask_on_blob_graph(data, attention_mask, title="Attention over Blob Graph", cmap='viridis'):
    """
    Draws a blob graph with nodes colored by their attention value (0 to 1).
    
    Args:
        data: PyG Data object
        attention_mask: Tensor of shape [num_nodes], with attention weights in [0, 1]
        title: plot title
        cmap: matplotlib colormap name
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    from torch_geometric.utils import to_networkx

    G = to_networkx(data, to_undirected=True)
    pos = nx.spring_layout(G, seed=42)  # consistent layout for the blob

    # Convert mask to numpy array
    attention = attention_mask.detach().cpu().numpy()
    attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)  # normalize

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(title)

    nodes = nx.draw_networkx_nodes(
        G, pos=pos,
        node_color=attention,
        cmap=plt.get_cmap(cmap),
        vmin=0, vmax=1,
        node_size=300,
        ax=ax
    )
    nx.draw_networkx_edges(G, pos=pos, edge_color='lightgray', ax=ax)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Attention Weight", rotation=270, labelpad=15)
    cbar.ax.tick_params(labelsize=8)

    ax.axis('off')
    plt.tight_layout()
    plt.savefig("attention_mask_blob.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    # coords = np.load('OCD_2_DATA/diffumo_coord_ROI512.npy') 
    graphs = torch.load("toy_data/toy_dataset_3ring_1000.pt")
    model_path = "model_paths_ASD/slepnet_best_run_0_batch_energy_sigmoid__toy_100.pth"  # Path to your trained model
    adj = graphs[0].edge_index
    adj = to_dense_adj(adj)[0]  # Convert edge index to dense adjacency matrix
     # Load cluster mask and model
    clusters_df = pd.read_csv("toy_data/three_ring_clusters.csv")
    # Convert the clusters to a dictionary
    clusters_dict = {col: clusters_df[col].dropna().tolist() for col in clusters_df.columns}
    clusters_dict = {k: [int(v) for v in vals] for k, vals in clusters_dict.items()}
    cluster_names = list(clusters_dict.keys())
    num_clusters = len(cluster_names)
    num_nodes = 150  
    # import pdb; pdb.set_trace()
    cluster_masks = torch.zeros(num_clusters, num_nodes)  # shape [num_clusters, num_nodes]

    for idx, cluster in enumerate(cluster_names):
        for node in clusters_dict[cluster]:
            cluster_masks[idx, node] = 1.0

    num_nodes = 150  # Number of nodes in AAL atlas
    num_features = 10  # Each node has 2 time-series features
    num_classes = 2   # Binary classification (single output per graph)
    num_clusters = 3  # Number of clusters
    # Load trained model
    model = SlepNet(
            num_nodes=num_nodes,
            num_features=num_features,
            num_clusters=num_clusters,
            cluster_masks=cluster_masks,
            num_slepians=100,
            num_classes=num_classes,
            layer_type="batch_energy",
            adj=adj
        )
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    # Compute this from your model
    with torch.no_grad():
        layer = model.sleplayer
        weights = torch.sigmoid(layer.cluster_attention)         # [num_clusters, 1]
        combined_mask = (weights.T @ cluster_masks).squeeze(0)          # [num_nodes = 512]
        thresholded_mask = (combined_mask > 0.7).float()
        # import pdb; pdb.set_trace() 
   
    topk = 50
    topk_indices = torch.topk(combined_mask, topk).indices.cpu().numpy()

    # Define inner ring node indices (0 to 49)
    middle_ring_nodes = set(range(50, 100))  # Middle ring is the informative region
    # inner_ring_nodes = set(range(0, 50))  # Inner ring is the informative region

    # Count how many of the top-k are in the informative set
    topk_set = set(topk_indices)
    overlap = topk_set & middle_ring_nodes
    num_overlap = len(overlap)

    # Output
    print("Top 50 node indices (by attention):")
    print(topk_indices.tolist())

    print(f"\nNumber of top-50 nodes in inner ring: {num_overlap} / {topk}")
    print(f"Overlap nodes: {sorted(list(overlap))}")
    # Draw the graph with highlighted nodes
    # draw_attention_highlight_ring_graph_full(graphs[0], combined_mask, title="Attention Weights on 3-Ring Graph", cmap='viridis')
    # draw_attention_highlight_er_graph_full(graphs[0], combined_mask, title="Top-50 Attention Nodes", cmap='viridis')
    draw_attention_mask_on_blob_graph(graphs[0], combined_mask, title="Attention Mask on Blob Graph", cmap='viridis')