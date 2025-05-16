import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from nilearn.datasets import fetch_abide_pcp, fetch_atlas_basc_multiscale_2015
from nilearn.input_data import NiftiLabelsMasker
from tqdm import tqdm
import math
import pandas as pd

def get_sinusoidal_embedding(timestep, dim, device='cpu'):
    half_dim = dim // 2
    freq_scale = torch.exp(
        -torch.arange(half_dim, dtype=torch.float, device=device) * math.log(10000.0) / (half_dim - 1)
    )
    angles = timestep * freq_scale
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=0)  # [dim]
    return emb  # shape: [dim]

# ---------------------- Step 1: Fetch ABIDE --------------------------
def fetch_abide_balanced():
    data_dir = '/gpfs/gibbs/pi/krishnaswamy_smita/sv496/nilearn_data/ABIDE_pcp/'

    # Load metadata only
    metadata = fetch_abide_pcp(
        n_subjects=None,
        derivatives=[],
        data_dir=data_dir,
        verbose=0
    )['phenotypic']

    # Sample 40 ASD and 40 HC
    asd = metadata[metadata['DX_GROUP'] == 1].head(40)
    hc = metadata[metadata['DX_GROUP'] == 2].head(40)
    selected = pd.concat([asd, hc])
    subject_ids = selected['SUB_ID'].tolist()

    # Fetch only those subjects using pre-downloaded files
    return fetch_abide_pcp(
        n_subjects=None,
        SUB_ID=subject_ids,
        pipeline='cpac',
        band_pass_filtering=True,
        global_signal_regression=True,
        derivatives=['func_preproc'],
        data_dir=data_dir,
        verbose=1
    )
def fetch_abide(n_subjects=5):
    return fetch_abide_pcp(
        n_subjects=n_subjects,
        pipeline='cpac',
        band_pass_filtering=True,
        global_signal_regression=True,
        derivatives=['func_preproc'],
        data_dir='/gpfs/gibbs/pi/krishnaswamy_smita/sv496/nilearn_data',
        verbose=0
    )

# ---------------------- Step 2: Extract time series -------------------
def extract_time_series(func_img, labels_img):
    masker = NiftiLabelsMasker(
        labels_img=labels_img,
        standardize=True,
        detrend=True,
        t_r=2.5
    )
    time_series = masker.fit_transform(func_img)  # shape: (T, N)
    return time_series.T  # shape: (T, N)

# ---------------------- Step 3: Compute FC adjacency from full series -
# def compute_fc_graph(time_series, threshold=0.3):
#     """time_series: (n_nodes, T)"""
#     fc = np.corrcoef(time_series)
#     # fc[np.isnan(fc)] = 0  # Handle NaN values
#     # import pdb; pdb.set_trace()
#     np.fill_diagonal(fc, 0)
#     fc[np.abs(fc) < threshold] = 0  # Sparsify weak edges
#     return fc
def compute_fc_graph(time_series, threshold=0.3):
    """
    time_series: shape (n_nodes, T)
    Returns: adjacency matrix of shape (n_nodes, n_nodes)
    """
    stds = time_series.std(axis=1)
    valid = stds > 1e-6

    if not valid.all():
        time_series = time_series[valid]

    fc = np.corrcoef(time_series)
    np.fill_diagonal(fc, 0)
    fc[np.abs(fc) < threshold] = 0
    return fc

# ---------------------- Step 4: Build PyG graph -----------------------
def build_graph(node_features, adj_matrix, time_emb=None):
    edge_index, edge_weight = dense_to_sparse(torch.tensor(adj_matrix, dtype=torch.float))

    x = torch.tensor(node_features, dtype=torch.float).unsqueeze(1)  # (N,) → (N, 1)

    if time_emb is not None:
        time_emb_expanded = time_emb.unsqueeze(0).repeat(x.size(0), 1)  # shape: [197, 16]
        x = torch.cat([x, time_emb_expanded], dim=1)  # final shape: [197, 17]

    return Data(x=x, edge_index=edge_index, edge_attr=edge_weight)


# ---------------------- Step 5: Create per-timepoint graphs ----------
def create_timepoint_graphs(abide_data):
    atlas = fetch_atlas_basc_multiscale_2015(version='sym')
    labels_img = atlas['scale197']  # 197-region parcellation

    # Grab phenotypic labels
    phenos = abide_data['phenotypic']  # list of dicts, same order as func_preproc
    all_graphs = []

    for i, func_img in enumerate(tqdm(abide_data['func_preproc'])):
        try:
            ts = extract_time_series(func_img, labels_img)  # shape: (T, 197)
            # import pdb; pdb.set_trace()
            adj = compute_fc_graph(ts, threshold=0.3)       # shape: (197, 197)
        except RuntimeWarning:
            print(f"RuntimeWarning encountered for {func_img}. Skipping.")
            continue

        # Get binary label from phenotype metadata
        pheno = phenos.iloc[i]
        dx_group = pheno.get("DX_GROUP", None)
        sex = pheno.get("SEX", None)
        print(f"DX_GROUP: {dx_group}")
        if dx_group not in [1, 2]:
            print(f"Skipping subject {i} due to missing DX_GROUP")
            continue
        label = 1 if dx_group == 1 else 0  # ASD = 1, HC = 0
        sex_label = 1 if sex == 1 else 0 # M = 1, F = 0

        time_emb_dim = 16
        for t in range(ts.shape[1]):
            node_feats = ts[:, t]  # shape: (197,)
            if time_emb_dim is not None:
                time_emb = get_sinusoidal_embedding(torch.tensor(t), dim=time_emb_dim)
                graph = build_graph(node_feats, adj, time_emb)
            else:
                graph = build_graph(node_feats, adj)
            graph.y = torch.tensor(label, dtype=torch.long)  # binary label per graph
            graph.sex = torch.tensor(sex_label, dtype=torch.long)  # binary label per graph
            graph.time = torch.tensor(t, dtype=torch.long)        # Timepoint index
            all_graphs.append(graph)

    return all_graphs

abide_data = fetch_abide_balanced()
graphs = create_timepoint_graphs(abide_data)
print(f"{len(graphs)} timepoint-level graphs created.")
print(graphs[0])

torch.save(graphs, 'abide_graphs_balanced.pt')
