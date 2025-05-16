import numpy as np
import torch
import pandas as pd
from sklearn.cluster import SpectralClustering

# === 1. Load the adjacency matrix ===
adj = np.load('PEMS07/adjacency_matrix.npy')  # shape [N, N]
num_clusters = 60

# === 2. Perform Spectral Clustering ===
sc = SpectralClustering(n_clusters=num_clusters, affinity='precomputed', assign_labels='kmeans', random_state=42)
cluster_assignments = sc.fit_predict(adj)  # shape: [N]

# === 3. Organize nodes per cluster ===
cluster_dict = {i: [] for i in range(num_clusters)}
for node, cluster in enumerate(cluster_assignments):
    cluster_dict[cluster].append(float(node))  # store as float to match your example format

# === 4. Convert to CSV Format ===
# Pad clusters to equal length for saving to CSV
max_len = max(len(nodes) for nodes in cluster_dict.values())
padded_clusters = {
    f'Cluster {i}': cluster_dict[i] + [np.nan] * (max_len - len(cluster_dict[i]))
    for i in range(num_clusters)
}

df = pd.DataFrame(padded_clusters)
df.to_csv('PEMS07/pems07_clusters.csv', index=False)
