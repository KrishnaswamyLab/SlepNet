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
from main_gnn import GCN, GIN, GAT, GraphSAGE
from src.GWT import GraphWaveletTransform  # import your GWT module
import os
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
            # H = model.sleplayer(g.x, g.edge_index, batch)  # (num_nodes, hidden_dim)
            # H = model.spectral_conv(g.x, g.edge_index, batch)  # (num_nodes, hidden_dim)
            H = model.encode(g.x, g.edge_index)  # (num_nodes, hidden_dim)
            if topk_nodes is not None:
                H = H[topk_nodes]

            pooled = H.mean(dim=0)  # (hidden_dim,)
            patient_embeddings.append(pooled.cpu().numpy())

    return np.stack(patient_embeddings)

def collect_gwt_embeddings_per_patient(graphs, indices, gwt):
    gwt.eval()
    patient_embeddings = []

    for idx in indices:
        g = graphs[idx]
        batch = torch.zeros(g.x.size(0), dtype=torch.long, device=g.x.device)

        gwt.X_init = g.x.to(gwt.device)
        gwt.edge_index = g.edge_index.to(gwt.device)
        gwt.edge_weight = torch.ones(g.edge_index.size(1)).to(gwt.device)

        with torch.no_grad():
            emb = gwt.generate_timepoint_features(batch)  # [1, feat_dim]
        patient_embeddings.append(emb.squeeze(0).cpu().numpy())

    return np.stack(patient_embeddings)


# def plot_patient_tphate_3d(embeddings, patient_label, patient_idx, output_dir, topk=False):
#     """
#     3D TPHATE plot for one patient.
#     """
#     tphate_operator = tphate.TPHATE(n_components=3)
#     emb_3d = tphate_operator.fit_transform(embeddings)

#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')

#     ax.scatter(emb_3d[:, 0], emb_3d[:, 1], emb_3d[:, 2], c=np.arange(len(emb_3d)), cmap='plasma', s=5, alpha=0.7)
#     ax.set_xlabel("TPHATE 1")
#     ax.set_ylabel("TPHATE 2")
#     ax.set_zlabel("TPHATE 3")
    
#     title_type = "Top-50 Nodes" if topk else "All Nodes"
#     ax.set_title(f"T-PHATE 3D Patient {patient_idx} ({'OCD' if patient_label==1 else 'HC'}) - {title_type}")

#     cbar = fig.colorbar(ax.scatter(emb_3d[:, 0], emb_3d[:, 1], emb_3d[:, 2], c=np.arange(len(emb_3d)), cmap='plasma'))
#     cbar.set_label('Timepoint')

#     plt.tight_layout()

#     filename = f"{output_dir}/tphate_3d_patient{patient_idx}_{'OCD' if patient_label==1 else 'HC'}_{'top50' if topk else 'allnodes'}_3D.png"
#     plt.savefig(filename, dpi=200)
#     plt.close()
#     print(f"Saved 3D plot to {filename}")

def obtain_tphate_embeddings(embeddings, patient_label, patient_idx, output_dir, topk=False):
    """
    3D TPHATE plot for one patient.
    """
    tphate_operator = tphate.TPHATE(n_components=3, verbose=0)
    emb_3d = tphate_operator.fit_transform(embeddings)
    return emb_3d
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')

    # ax.scatter(emb_3d[:, 0], emb_3d[:, 1], emb_3d[:, 2], c=np.arange(len(emb_3d)), cmap='plasma', s=5, alpha=0.7)
    # ax.set_xlabel("TPHATE 1")
    # ax.set_ylabel("TPHATE 2")
    # ax.set_zlabel("TPHATE 3")
    
    # title_type = "Top-50 Nodes" if topk else "All Nodes"
    # ax.set_title(f"T-PHATE 3D Patient {patient_idx} ({'OCD' if patient_label==1 else 'HC'}) - {title_type}")

    # cbar = fig.colorbar(ax.scatter(emb_3d[:, 0], emb_3d[:, 1], emb_3d[:, 2], c=np.arange(len(emb_3d)), cmap='plasma'))
    # cbar.set_label('Timepoint')

    # plt.tight_layout()

    # filename = f"{output_dir}/tphate_3d_patient{patient_idx}_{'OCD' if patient_label==1 else 'HC'}_{'top50' if topk else 'allnodes'}_3D.png"
    # plt.savefig(filename, dpi=200)
    # plt.close()
    # print(f"Saved 3D plot to {filename}")

# --- Load everything ---

# coords = np.load('OCD_data/diffumo_coord_ROI512.npy')
graphs = torch.load("OCD_data/all_graphs_MF.pt")
model_path = "gnn_paths_ASD/GCN_best_run_0_ASD.pth"

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
# model = SlepNet(
#     num_nodes=num_nodes,
#     num_features=17,
#     num_clusters=num_clusters,
#     cluster_masks=cluster_masks,
#     num_slepians=100,
#     num_classes=2,
#     layer_type="batch_energy",
#     adj=adj
# )

# model = SpectralNet(
#             num_nodes=num_nodes,
#             num_features=17,
#             num_classes=2,
#             hidden_dim=250,
#             num_eigenvectors=10,
#             adj=adj
#         )

# model = GCN(num_features=17, hidden_dims=[128,64], num_classes=2)
# model = GIN(num_features=17, hidden_dims=[128,64], num_classes=2)
# model = GAT(num_features=17, hidden_dims=[128,64], num_classes=2)
# model = GraphSAGE(num_features=17, hidden_dims=[128,64], num_classes=2)
dummy_graph = graphs[0]
gwt = GraphWaveletTransform(
    edge_index=dummy_graph.edge_index,
    edge_weight=torch.ones(dummy_graph.edge_index.size(1)),
    X=dummy_graph.x,
    J=3,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# checkpoint = torch.load(model_path, map_location='cpu')
# model.load_state_dict(checkpoint['model_state_dict'])

# --- Split patients ---
labels = np.array([g.y.item() for g in graphs])
timepoints = np.array([g.time for g in graphs])
genders = np.array([g.gender for g in graphs])  # Extract gender labels

# Collect graphs patient-wise
patients = {}
current_patient = []
current_label = labels[0]
current_gender = genders[0]
for i in range(len(graphs)):
    if timepoints[i] == 0 and len(current_patient) > 0:
        # Save the previous patient
        patients[len(patients)] = (current_patient, current_label, current_gender)
        current_patient = []
        current_label = labels[i]
        current_gender = genders[i]
    current_patient.append(i)
# Save last patient
patients[len(patients)] = (current_patient, current_label, current_gender)

# Pick 5 OCD, 5 HC patients
ocd_patients = [pid for pid, (idxs, label, gender) in patients.items() if label == 1]
hc_patients = [pid for pid, (idxs, label, gender) in patients.items() if label == 0]

selected_patients = ocd_patients + hc_patients

# --- Compute top-50 nodes ---
# with torch.no_grad():
#     layer = model.sleplayer
#     weights = torch.sigmoid(layer.cluster_attention)
#     combined_mask = (weights.T @ cluster_masks).squeeze(0)
#     top50_nodes = torch.topk(combined_mask, k=50).indices.cpu()

# --- Create output directory ---
output_dir = "tphate_embeddings_final_PVDM"
os.makedirs(output_dir, exist_ok=True)

# --- Collect TPHATE embeddings for each patient ---
tphate_embeddings_chunks = []
gender_labels = []

for pid in tqdm(selected_patients):
    graph_indices, label, gender = patients[pid]
    # Collect embeddings for all nodes
    # emb_all = collect_embeddings_per_patient(model, graphs, graph_indices, topk_nodes=None)
    emb_all = collect_gwt_embeddings_per_patient(graphs, graph_indices, gwt)

    tphate_embeddings = obtain_tphate_embeddings(emb_all, label, pid, output_dir, topk=False)  # Shape: (num_timepoints, 3)
    # Check if the number of timepoints is divisible by 20
    if tphate_embeddings.shape[0] % 20 == 0:
        # Reshape into chunks of 20 and append to the list
        segmented_embeddings = tphate_embeddings.reshape(-1, 20, 3)  # Shape: (num_chunks, 20, 3)
        tphate_embeddings_chunks.extend(segmented_embeddings)
        
        # Replicate the gender label 20 times for each chunk
        gender_labels.extend([gender] * segmented_embeddings.shape[0])
    else:
        print(f"Skipping patient {pid} due to non-divisible timepoints: {tphate_embeddings.shape[0]}")
# tphate_embeddings_chunks = []
# gender_labels = []

# for pid in tqdm(selected_patients):
#     graph_indices, label, gender = patients[pid]
#     emb_all = collect_embeddings_per_patient(model, graphs, graph_indices, topk_nodes=None)
#     tphate_embeddings = obtain_tphate_embeddings(emb_all, label, pid, output_dir, topk=False)  # (T, 3)

#     # NEW: Truncate to nearest multiple of 20
#     T = tphate_embeddings.shape[0]
#     usable_len = (T // 20) * 20  # e.g., 196 → 180

#     if usable_len == 0:
#         print(f"Skipping patient {pid} — not enough timepoints")
#         continue

#     truncated = tphate_embeddings[:usable_len]
#     segmented = truncated.reshape(-1, 20, 3)  # Shape: (usable_len // 20, 20, 3)

#     tphate_embeddings_chunks.extend(segmented)
#     gender_labels.extend([gender] * segmented.shape[0])
    ################ Append full time series embedding and its label ################
    # tphate_embeddings_chunks.append(tphate_embeddings)  # shape: (T, 3)
    # gender_labels.append(gender)

# import pdb; pdb.set_trace()
# Save the embeddings and labels
np.save(os.path.join(output_dir, "gwt_embeddings.npy"), tphate_embeddings_chunks)
np.save(os.path.join(output_dir, "gwt_gender_labels.npy"), gender_labels)
# import pdb; pdb.set_trace()

# --- Load the saved embeddings and labels ---
output_dir = "tphate_embeddings_final_PVDM"
tphate_embeddings_chunks = np.load(os.path.join(output_dir, "gwt_embeddings.npy"), allow_pickle=True)
gender_labels = np.load(os.path.join(output_dir, "gwt_gender_labels.npy"), allow_pickle=True)
# --- Train an MLP to predict gender ---
import torch.nn as nn
import torch.optim as optim

# Convert data to tensors
X = torch.tensor(tphate_embeddings_chunks, dtype=torch.float32)  # Shape: (num_samples, 20, 3)
y = torch.tensor(gender_labels, dtype=torch.long)  # Shape: (num_samples,)

# Flatten the input features
X = X.view(X.size(0), -1)  # Shape: (num_samples, 60)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Perform 10 runs
num_runs = 10
num_epochs = 300
all_accuracies = []

for run in range(num_runs):
    print(f"Run {run + 1}/{num_runs}")

    # Initialize the model, loss function, and optimizer
    input_dim = X_train.size(1)
    hidden_dim = 128
    output_dim = 2  # Binary classification (gender)
    model = MLP(input_dim, hidden_dim, output_dim)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train.squeeze())
        loss.backward()
        optimizer.step()

        # Evaluate on the test set
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test).argmax(dim=1)
            accuracy = accuracy_score(y_test.numpy(), y_pred.numpy())

        # Save the best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            # torch.save(model.state_dict(), os.path.join(output_dir, f"best_mlp_model_run{run + 1}.pth"))

        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Test Accuracy: {accuracy:.4f}")

    print(f"Run {run + 1} Best Test Accuracy: {best_accuracy:.4f}")
    all_accuracies.append(best_accuracy)

# Print overall results
print(f"All Test Accuracies: {all_accuracies}")
print(f"Mean Test Accuracy: {np.mean(all_accuracies):.4f}, Std Dev: {np.std(all_accuracies):.4f}")
