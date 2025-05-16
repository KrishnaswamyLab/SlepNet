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
from torch_geometric.data import DataLoader
from main_gnn import GCN, GIN, GAT, GraphSAGE, TransformerGNN
from src.SpecNet import SpectralNet
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser
import gc
from tqdm import tqdm
import wandb

gc.enable()

# -------------------------------
# Argument Parsing for Training
# -------------------------------
parser = ArgumentParser(description="Graph Neural Network Classifier")
parser.add_argument('--raw_dir', type=str, default='gnn_paths_PVDM', help="Directory where the data is stored")
parser.add_argument('--task', type=str, default='classification', help="Classification task type")
parser.add_argument('--hidden_dims', type=int, nargs='+', default=[128,64], help="Hidden dimensions per layer")
parser.add_argument('--hidden_dim', type=int, default=250, help="Hidden dim for spectral MLP")
parser.add_argument('--num_eigenvectors', type=int, default=100, help="Number of eigenvectors")
parser.add_argument('--lr', type=float, default=1e-3, help="Learning Rate")
parser.add_argument('--wd', type=float, default=1e-2, help="Weight decay")
parser.add_argument('--num_epochs', type=int, default=100, help="Number of epochs")
parser.add_argument('--batch_size', type=int, default=128, help="Batch size")
parser.add_argument('--gpu', type=int, default=0, help="GPU index")
parser.add_argument('--model', type=str, default='SpectralNet',
                    choices=['GCN', 'GIN', 'GAT', 'GraphSAGE', 'TransformerGNN', 'SpectralNet'],
                    help="GNN model type")
parser.add_argument('--dataset', type=str, default='pvdm', choices=['abide', 'pvdm', 'ra'], help="Dataset to use") 
parser.add_argument('--num_runs', type=int, default=10, help="Number of training runs")
parser.add_argument('--device', type=str, default='cuda', help="Device to run the model on")
args = parser.parse_args()
# -------------------------------
# Evaluation Function
# -------------------------------
def test(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(args.device)
            out = model(batch.x, batch.edge_index, batch.batch)#.squeeze(1)
            preds = out.argmax(dim=1)
            # import pdb; pdb.set_trace()
            correct += torch.sum(preds == batch.y).float()
            total += batch.y.size(0)
    # import pdb; pdb.set_trace()
    return (correct * 100) / total

# -------------------------------
# Training Function
# -------------------------------
def train(model, run_id):
    print(f"Starting Run {run_id + 1}/{args.num_runs}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    loss_fn = torch.nn.CrossEntropyLoss()
    best_acc = 0

    with tqdm(range(args.num_epochs)) as tq:
        for epoch in tq:
            model.train()
            correct_train = 0
            t_loss = 0

            for batch in train_loader:
                batch = batch.to(args.device)
                opt.zero_grad()
                out = model(batch.x, batch.edge_index, batch.batch)#.squeeze(1)
                loss = loss_fn(out, batch.y)
                loss.backward()
                opt.step()

                preds = out.argmax(dim=1)
                correct_train += torch.sum(preds == batch.y).float()
                t_loss += loss.item()

            train_acc = (correct_train * 100) / len(train_graphs)
            test_acc = test(model, test_loader)

            wandb.log({'Run': run_id, 'Loss': t_loss, 'Train acc': train_acc, 'Test acc': test_acc}, step=epoch+1)

            if test_acc > best_acc:
                best_acc = test_acc
                model_path = f"{args.raw_dir}/{args.model}_best_run_{run_id}_PVDM.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'best_acc': best_acc,
                    'args': args
                }, model_path)

            tq.set_description(f"[Run {run_id+1}] Train acc = {train_acc:.4f}, Test acc = {test_acc:.4f}, Best acc = {best_acc:.4f}")

    print(f"Best accuracy for Run {run_id + 1}: {best_acc}")
    return best_acc

if __name__ == "__main__":
    if args.dataset == 'abide':
        print("Loading graph dataset...")
        graph_dataset = torch.load("BioPoint/abide_graphs_balanced.pt")  # List of PyG Data objects
        model_path = "model_paths_ASD/slepnet_best_run_0_batch_energy_sigmoid__abide_100.pth"
        # with open(f"{args.raw_dir}/fnirs_graphs_deoxy.pkl", 'rb') as f:
        #     graph_dataset = pickle.load(f)
        # Normalize node features in all graphs
        adj = graph_dataset[0].edge_index
        adj = to_dense_adj(adj)[0]
        # Count the number of graphs with label 0 and 1
        # label_counts = {0: 0, 1: 0}
        # for graph in graph_dataset:
        #     label_counts[int(graph.y.item())] += 1

        # print(f"Number of graphs with label 0: {label_counts[0]}")
        # print(f"Number of graphs with label 1: {label_counts[1]}")
        # import pdb; pdb.set_trace()
        # for graph in graph_dataset:
        #     graph.x = (graph.x - graph.x.mean(dim=0)) / (graph.x.std(dim=0) + 1e-6)   # Avoid division by zero
        ####Load the clusters####
        # Load the clusters from the CSV file
        # import pdb; pdb.set_trace()
        clusters_df = pd.read_csv("BioPoint/node_cluster_assignments_abide.csv")
        # Convert the clusters to a dictionary
        clusters_dict = {col: clusters_df[col].dropna().tolist() for col in clusters_df.columns}
        clusters_dict = {k: [int(v) for v in vals] for k, vals in clusters_dict.items()}
        cluster_names = list(clusters_dict.keys())
        num_clusters = len(cluster_names)
        num_nodes = 197  
        # import pdb; pdb.set_trace()
        cluster_masks = torch.zeros(num_clusters, num_nodes)  # shape [num_clusters, num_nodes]

        for idx, cluster in enumerate(cluster_names):
            for node in clusters_dict[cluster]:
                cluster_masks[idx, node] = 1.0
    elif args.dataset == 'pvdm':
        print("Loading graph dataset...")
        graph_dataset = torch.load("OCD_data/all_graphs_time_emb.pt")  # List of PyG Data objects
        model_path = "model_paths_ASD/slepnet_best_run_0_batch_energy_sigmoid__pvdm_100.pth"
        # with open(f"{args.raw_dir}/fnirs_graphs_deoxy.pkl", 'rb') as f:
        #     graph_dataset = pickle.load(f)
        # Normalize node features in all graphs
        adj = graph_dataset[0].edge_index
        adj = to_dense_adj(adj)[0]
        # Count the number of graphs with label 0 and 1
        # label_counts = {0: 0, 1: 0}
        # for graph in graph_dataset:
        #     label_counts[int(graph.y.item())] += 1

        # print(f"Number of graphs with label 0: {label_counts[0]}")
        # print(f"Number of graphs with label 1: {label_counts[1]}")
        # import pdb; pdb.set_trace()
        # for graph in graph_dataset:
        #     graph.x = (graph.x - graph.x.mean(dim=0)) / (graph.x.std(dim=0) + 1e-6)   # Avoid division by zero
        ####Load the clusters####
        # Load the clusters from the CSV file
        # import pdb; pdb.set_trace()
        clusters_df = pd.read_csv("OCD_data/node_cluster_assignments.csv")
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
    elif args.dataset == 'ra':
        print("Loading graph dataset...")
        graph_dataset = torch.load("OCD_2_DATA/all_graphs_time_emb_RA.pt")  # List of PyG Data objects
        model_path = "model_paths_ASD/slepnet_best_run_0_batch_energy_sigmoid__ra_100.pth"
        # with open(f"{args.raw_dir}/fnirs_graphs_deoxy.pkl", 'rb') as f:
        #     graph_dataset = pickle.load(f)
        # Normalize node features in all graphs
        adj = graph_dataset[0].edge_index
        adj = to_dense_adj(adj)[0]
        # Count the number of graphs with label 0 and 1
        # label_counts = {0: 0, 1: 0}
        # for graph in graph_dataset:
        #     label_counts[int(graph.y.item())] += 1

        # print(f"Number of graphs with label 0: {label_counts[0]}")
        # print(f"Number of graphs with label 1: {label_counts[1]}")
        # import pdb; pdb.set_trace()
        # for graph in graph_dataset:
        #     graph.x = (graph.x - graph.x.mean(dim=0)) / (graph.x.std(dim=0) + 1e-6)   # Avoid division by zero
        ####Load the clusters####
        # Load the clusters from the CSV file
        # import pdb; pdb.set_trace()
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

    # num_nodes = 512  # Number of nodes in AAL atlas
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
        topk_indices = torch.topk(combined_mask, k=50).indices


    # Prepare the dataset with top 50 nodes
    topk_nodes = topk_indices.cpu().numpy()
    filtered_graphs = []
    labels = []

    for graph in graph_dataset:
        edge_index = graph.edge_index
        node_features = graph.x
        label = graph.y

        # Filter the graph to include only top 50 nodes
        mask = torch.zeros(node_features.size(0), dtype=torch.bool)
        mask[topk_nodes] = True
        filtered_features = node_features[mask]

        # Create lookup from old to new index
        id_map = {old_idx: new_idx for new_idx, old_idx in enumerate(topk_nodes)}
        edge_mask = mask[edge_index[0]] & mask[edge_index[1]]
        filtered_edge_index = edge_index[:, edge_mask]

        # Remap edge indices
        remapped_edge_index = torch.stack([
            torch.tensor([id_map[int(i)] for i in filtered_edge_index[0]]),
            torch.tensor([id_map[int(i)] for i in filtered_edge_index[1]])
        ], dim=0)

        # Create a new graph
        filtered_graph = Data(x=filtered_features, edge_index=remapped_edge_index, y=label)
        filtered_graphs.append(filtered_graph)



    all_best_accs = []

    for run_id in range(10):
        train_idx, test_idx = train_test_split(np.arange(len(filtered_graphs)), test_size=0.2, random_state=run_id)
        train_graphs = [filtered_graphs[i] for i in train_idx]
        test_graphs = [filtered_graphs[i] for i in test_idx]

        train_loader = DataLoader(train_graphs, batch_size=128, shuffle=True)
        test_loader = DataLoader(test_graphs, batch_size=128)

        wandb.init(project=f'{args.model}-training', name=f"run_{run_id}", config=vars(args), reinit=True)
        
        # model_classes = {'GCN': GCN, 'GIN': GIN, 'GAT': GAT, 'GraphSAGE': GraphSAGE, 'TransformerGNN': TransformerGNN}
        # model = model_classes[args.model](num_features=17, hidden_dims=args.hidden_dims, num_classes=2).to(args.device)
        num_nodes = 50
        num_features = train_graphs[0].x.size(1)
        num_classes = 2
        # model = SpectralNet(
        #     num_nodes=num_nodes,
        #     num_features=num_features,
        #     num_classes=num_classes,
        #     hidden_dim=args.hidden_dim,
        #     num_eigenvectors=args.num_eigenvectors,
        #     adj=adj
        # ).to(args.device)
        model_classes = {
        'GCN': GCN,
        'GIN': GIN,
        'GAT': GAT,
        'GraphSAGE': GraphSAGE,
        'TransformerGNN': TransformerGNN
        }

        # Initialize model
        if args.model == 'SpectralNet':
            model = SpectralNet(
                num_nodes=num_nodes,
                num_features=num_features,
                num_classes=num_classes,
                hidden_dim=args.hidden_dim,
                num_eigenvectors=args.num_eigenvectors,
                adj=adj
            ).to(args.device)
        else:
            model = model_classes[args.model](
                num_features=num_features,
                hidden_dims=args.hidden_dims,
                num_classes=num_classes
            ).to(args.device)
        best_acc = train(model, run_id)
        all_best_accs.append(best_acc)
        mean_acc = np.mean([acc.cpu().item() for acc in all_best_accs])
        std_acc = np.std([acc.cpu().item() for acc in all_best_accs])
    print(f"Mean Best Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")