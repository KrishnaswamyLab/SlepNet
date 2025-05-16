import torch
import numpy as np
import tphate
import matplotlib.pyplot as plt
import pandas as pd
from torch_geometric.utils import to_dense_adj
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm
from src.SlepNet import SlepNet
from src.SpecNet import SpectralNet
import os
from mpl_toolkits.mplot3d import Axes3D

def collect_embeddings_per_patient(model, graphs, indices, topk_nodes=None):
    """
    Collect graph-level embeddings for graphs belonging to a single patient.
    
    Args:
        model: trained SlepNet
        graphs: list of all graphs
        indices: list of indices for graphs of this patient
        topk_nodes: list of top-k nodes (optional)

    Returns:
        (num_timepoints, hidden_dim) numpy array
    """
    model.eval()
    patient_embeddings = []

    for idx in indices:
        g = graphs[idx]
        with torch.no_grad():
            batch = torch.zeros(g.x.size(0), dtype=torch.long, device=g.x.device)
            H = model.sleplayer(g.x, g.edge_index, batch)  # (num_nodes, hidden_dim)
            # H = model.spectral_conv(g.x, g.edge_index, batch)  # (num_nodes, hidden_dim)
            if topk_nodes is not None:
                H = H[topk_nodes]
            # import pdb; pdb.set_trace()
            pooled = H.mean(dim=0)  # (hidden_dim,)
            patient_embeddings.append(pooled.cpu().numpy())

    return np.stack(patient_embeddings)

def plot_patient_tphate_3d(embeddings, patient_label, patient_idx, output_dir, topk=False):
    """
    3D TPHATE plot for one patient with colorbar included.
    """
    tphate_operator = tphate.TPHATE(n_components=3)
    emb_3d = tphate_operator.fit_transform(embeddings)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(emb_3d[:, 0], emb_3d[:, 1], emb_3d[:, 2], c=np.arange(len(emb_3d)), cmap='plasma', s=100, alpha=0.7)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # Add colorbar to the same plot
    # cbar = fig.colorbar(scatter, ax=ax, orientation='vertical', pad=0.1)
    # cbar.set_label('Timepoint', fontsize=18)
    # cbar.ax.tick_params(labelsize=18)  # Increase font size of the numbers on the colorbar

    plt.tight_layout()

    # Save the 3D plot
    filename = f"{output_dir}/tphate_3d_patient{patient_idx}_{'OCD' if patient_label==1 else 'HC'}_{'top50' if topk else 'allnodes'}_3D_NOBAR.png"
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"Saved 3D plot to {filename}")



def get_tphate_embeddings(embeddings, patient_label, patient_idx, output_dir, topk=False):
    """
    3D TPHATE plot for one patient with colorbar included.
    """
    tphate_operator = tphate.TPHATE(n_components=3)
    emb_3d = tphate_operator.fit_transform(embeddings)

    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')

    # scatter = ax.scatter(emb_3d[:, 0], emb_3d[:, 1], emb_3d[:, 2], c=np.arange(len(emb_3d)), cmap='plasma', s=100, alpha=0.7)

    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])

    # # Add colorbar to the same plot
    # # cbar = fig.colorbar(scatter, ax=ax, orientation='vertical', pad=0.1)
    # # cbar.set_label('Timepoint', fontsize=18)
    # # cbar.ax.tick_params(labelsize=18)  # Increase font size of the numbers on the colorbar

    # plt.tight_layout()

    # # Save the 3D plot
    # filename = f"{output_dir}/tphate_3d_patient{patient_idx}_{'OCD' if patient_label==1 else 'HC'}_{'top50' if topk else 'allnodes'}_3D_NOBAR.png"
    # plt.savefig(filename, dpi=200)
    # plt.close()
    # print(f"Saved 3D plot to {filename}")
    return emb_3d


# --- Load everything ---

# coords = np.load('OCD_2_DATA/diffumo_coord_ROI512.npy')
graphs = torch.load("OCD_data/all_graphs_time_emb.pt")
model_path = "model_paths_ASD/slepnet_best_run_0_batch_energy_sigmoid__pvdm_100.pth"
# import pdb; pdb.set_trace()
# Cluster masks
clusters_df = pd.read_csv("OCD_data/node_cluster_assignments.csv")
clusters_dict = {col: clusters_df[col].dropna().tolist() for col in clusters_df.columns}
clusters_dict = {k: [int(v) for v in vals] for k, vals in clusters_dict.items()}
num_clusters = len(clusters_dict)
num_nodes = 512
cluster_masks = torch.zeros(num_clusters, num_nodes)
for idx, cluster in enumerate(clusters_dict.keys()):
    for node in clusters_dict[cluster]:
        cluster_masks[idx, node] = 1.0

# Adjacency
adj = graphs[0].edge_index
adj = to_dense_adj(adj)[0]

# Model
model = SlepNet(
    num_nodes=num_nodes,
    num_features=17,
    num_clusters=num_clusters,
    cluster_masks=cluster_masks,
    num_slepians=100,
    num_classes=2,
    layer_type="batch_energy",
    adj=adj
)

# model = SpectralNet(
#             num_nodes=num_nodes,
#             num_features=17,
#             num_classes=2,
#             hidden_dim=250,
#             num_eigenvectors=10,
#             adj=adj
#         )
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])

# --- Split patients ---
labels = np.array([g.y.item() for g in graphs])
timepoints = np.array([g.time for g in graphs])

# Collect graphs patient-wise
patients = {}
current_patient = []
current_label = labels[0]
for i in range(len(graphs)):
    if timepoints[i] == 0 and len(current_patient) > 0:
        # Save the previous patient
        patients[len(patients)] = (current_patient, current_label)
        current_patient = []
        current_label = labels[i]
    current_patient.append(i)
# Save last patient
patients[len(patients)] = (current_patient, current_label)

# Pick 5 OCD, 5 HC patients
asd_patients = [pid for pid, (idxs, label) in patients.items() if label == 1]
hc_patients = [pid for pid, (idxs, label) in patients.items() if label == 0]

selected_patients = asd_patients + hc_patients

# --- Compute top-50 nodes ---
# with torch.no_grad():
#     layer = model.sleplayer
#     weights = torch.sigmoid(layer.cluster_attention)
#     combined_mask = (weights.T @ cluster_masks).squeeze(0)
#     top50_nodes = torch.topk(combined_mask, k=50).indices.cpu()

# --- Create output directory ---
output_dir = "TPHATE_embeddings_final_energy_PVDM"
os.makedirs(output_dir, exist_ok=True)

# --- Plot for each patient ---
ocd_tphate_embeddings = []
hc_tphate_embeddings = []
for pid in selected_patients:
    graph_indices, label = patients[pid]

    # (1) All nodes
    emb_all = collect_embeddings_per_patient(model, graphs, graph_indices, topk_nodes=None)
    # raw_data = np.array([graphs[idx].x.cpu().numpy() for idx in graph_indices])  # Collect raw node features
    # # import pdb; pdb.set_trace()
    # raw_data = torch.tensor(raw_data).float()
    # raw_data = raw_data.mean(dim=1)
    # import pdb; pdb.set_trace()
    emb_all = get_tphate_embeddings(emb_all, label, pid, output_dir, topk=False)  # Directly compute TPHATE
    # plot_patient_tphate_3d(emb_all, label, pid, output_dir, topk=False)
    # import pdb; pdb.set_trace()
    if emb_all.shape[0] != 420:
        print(f"Warning: Patient {pid} has {emb_all.shape[0]} timepoints instead of 420. Skipping...")
        continue
    if label == 1:
        ocd_tphate_embeddings.append(emb_all)
    else:
        hc_tphate_embeddings.append(emb_all)

    # import pdb; pdb.set_trace()
    # (2) Top-50 nodes only
    # emb_top50 = collect_embeddings_per_patient(model, graphs, graph_indices, topk_nodes=top50_nodes)
    # plot_patient_tphate_3d(emb_top50, label, pid, output_dir, topk=True)

# Combine the lists of numpy arrays into a single numpy array
ocd_tphate_embeddings = np.stack(ocd_tphate_embeddings)  # Shape: (len(ocd_tphate_embeddings), 420, 3)
hc_tphate_embeddings = np.stack(hc_tphate_embeddings)    # Shape: (len(hc_tphate_embeddings), 420, 3)
# import pdb; pdb.set_trace()
# # Save the combined arrays
np.save(f"{output_dir}/ocd_tphate_embeddings_energy_PVDM.npy", ocd_tphate_embeddings)
np.save(f"{output_dir}/hc_tphate_embeddings_energy_PVDM.npy", hc_tphate_embeddings)
# np.save(f"{output_dir}/ocd_tphate_embeddings_raw_RA.npy", np.concatenate(ocd_tphate_embeddings))
# np.save(f"{output_dir}/hc_tphate_embeddings_raw_RA.npy", np.concatenate(hc_tphate_embeddings))