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
def plot_cluster_mask_glass_brain_mesh(combined_mask, coords, output_path='BRAIN_MASK_RA_final_distance.png', cmap='plasma'):
    """
    Projects the SlepNet attention weights into a 3D brain volume and visualizes it as a mesh (not just points).
    
    Args:
        combined_mask (Tensor or array): shape (N,)
        coords (array): shape (N, 3) (MNI-space coordinates)
        output_path (str): File to save the brain mesh visualization
        cmap (str): Matplotlib colormap
    """
    # Convert input
    combined_mask = combined_mask.cpu().numpy() if isinstance(combined_mask, torch.Tensor) else combined_mask
    combined_mask = (combined_mask - np.min(combined_mask)) / (np.max(combined_mask) - np.min(combined_mask))

    assert coords.shape[0] == len(combined_mask), "Coordinate and mask mismatch"
    # import pdb; pdb.set_trace()
    # # Load MNI template
    # template_img = load_mni152_template(resolution=2)  # 2mm resolution, shape (91, 109, 91)
    # template_data = template_img.get_fdata()
    # affine = template_img.affine
    difumo_atlas = datasets.fetch_atlas_difumo(dimension=512, resolution_mm=2, legacy_format=False)

    atlas_img = nib.load(difumo_atlas.maps)  # Load the atlas as a NIfTI image
    atlas_data = atlas_img.get_fdata()
    affine = atlas_img.affine

    # Initialize empty volume
    brain_volume = np.zeros(atlas_data.shape[:3])


    # Map coordinates to voxel space
    from nilearn.image.resampling import coord_transform
    for i, (x, y, z) in enumerate(coords):
        vx, vy, vz = coord_transform(x, y, z, np.linalg.inv(affine))
        vx, vy, vz = int(round(vx)), int(round(vy)), int(round(vz))
        if 0 <= vx < brain_volume.shape[0] and 0 <= vy < brain_volume.shape[1] and 0 <= vz < brain_volume.shape[2]:
            brain_volume[vx, vy, vz] = combined_mask[i]
            # brain_volume[vz, vy, vx] = combined_mask[i]
    # from nilearn.image import coord_to_index

    # vox_indices = coord_to_index(coords, affine)  # returns (N, 3) voxel indices
    # for i, (i_, j_, k_) in enumerate(vox_indices):
    #     if (
    #         0 <= i_ < brain_volume.shape[0] and
    #         0 <= j_ < brain_volume.shape[1] and
    #         0 <= k_ < brain_volume.shape[2]
    #     ):
    #         brain_volume[i_, j_, k_] = combined_mask[i]

    # Apply slight Gaussian smoothing for visualization
    brain_volume = gaussian_filter(brain_volume, sigma=2)

    # Create a NIfTI image
    parcellated_signal_img = nib.Nifti1Image(brain_volume, affine)
    # nib.save(parcellated_signal_img, "slepnet_energy_RA_mask.nii.gz")
    # Visualize on the brain mesh
    display = plotting.plot_glass_brain(
        parcellated_signal_img,
        display_mode='lyrz',
        threshold=None,
        colorbar=True,
        cmap=cmap,
        title="SlepNet Node Attention Mesh"
    )
    display.savefig(output_path)
    display.close()
    print(f"Saved mesh visualization to {output_path}")
def plot_cluster_mask_glass_brain(combined_mask, coords, output_path='glass_brain_mask_with_colorbar_energy_7.png', cmap='plasma'):
    """
    Visualizes the learned node-level attention weights or cluster mask on a glass brain with a colorbar.

    Args:
        combined_mask (Tensor or array): shape (N,), e.g., from softmax-weighted cluster attention
        coords (array): shape (N, 3), MNI coordinates of the nodes
        output_path (str): file to save the image
        cmap (str): matplotlib colormap name
    """
    # Convert input
    combined_mask = combined_mask.cpu().numpy() if isinstance(combined_mask, torch.Tensor) else combined_mask
    combined_mask = (combined_mask - combined_mask.min()) / (combined_mask.max() - combined_mask.min())
    assert coords.shape[0] == len(combined_mask), "Coordinate and mask mismatch"

    # Colormap
    colormap = plt.get_cmap(cmap)
    node_colors = [colormap(val) for val in combined_mask]

    # Plot the brain
    display = plotting.plot_glass_brain(
        None,
        display_mode='lzry',
        title='Learned Cluster Mask (SlepNet)',
        plot_abs=False,
        colorbar=False,
        alpha=0.3
    )
    display.add_markers(coords, marker_color=node_colors, marker_size=6)

    # Create a separate figure just for the colorbar
    fig, ax = plt.subplots(figsize=(1.5, 5))
    norm = plt.Normalize(vmin=0, vmax=1)
    fig.subplots_adjust(left=0.3, right=0.6)
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax)
    cbar.set_label('Attention Weight', rotation=270, labelpad=15)
    cbar.ax.tick_params(labelsize=8)
    colorbar_path = output_path.replace('.png', '_colorbar.png')
    fig.savefig(colorbar_path, dpi=150)
    plt.close(fig)

    # # Save glass brain
    display.savefig(output_path)
    display.close()

    print(f"Saved glass brain plot to {output_path}")
    print(f"Saved colorbar separately to {colorbar_path}")


if __name__ == "__main__":
    coords = np.load('OCD_2_DATA/diffumo_coord_ROI512.npy') 
    graphs = torch.load("OCD_2_DATA/all_graphs_time_emb_RA.pt")
    model_path = "model_paths_ASD/slepnet_best_run_0_batch_distance_sigmoid__ra_100.pth"  # Path to your trained model
    adj = graphs[0].edge_index
    adj = to_dense_adj(adj)[0]  # Convert edge index to dense adjacency matrix
     # Load cluster mask and model
    clusters_df = pd.read_csv("OCD_2_DATA/node_cluster_assignments_RA.csv")
    # Convert the clusters to a dictionary
    clusters_dict = {col: clusters_df[col].dropna().tolist() for col in clusters_df.columns}
    clusters_dict = {k: [int(v) for v in vals] for k, vals in clusters_dict.items()}
    cluster_names = list(clusters_dict.keys())
    num_clusters = len(cluster_names)
    num_nodes = 512  
    # import pdb; pdb.set_trace()
    cluster_masks = torch.zeros(num_clusters, num_nodes)  # shape [num_clusters, num_nodes]

    for idx, cluster in enumerate(cluster_names):
        for node in clusters_dict[cluster]:
            cluster_masks[idx, node] = 1.0

    num_nodes = 512  # Number of nodes in AAL atlas
    num_features = 17  # Each node has 2 time-series features
    num_classes = 2   # Binary classification (single output per graph)
    num_clusters = 60  # Number of clusters
    # Load trained model
    model = SlepNet(
            num_nodes=num_nodes,
            num_features=num_features,
            num_clusters=num_clusters,
            cluster_masks=cluster_masks,
            num_slepians=100,
            num_classes=num_classes,
            layer_type="batch_distance",
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
    # import pdb; pdb.set_trace()
    plot_cluster_mask_glass_brain_mesh(thresholded_mask, coords)
    # plot_cluster_mask_glass_brain(combined_mask, coords)
