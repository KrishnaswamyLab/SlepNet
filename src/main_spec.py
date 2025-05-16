import torch
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
import wandb
import gc
from argparse import ArgumentParser
from src.SpecNet import SpectralNet
import pandas as pd
import pickle
from torch_geometric.utils import to_dense_adj

gc.enable()

# -------------------------------
# Argument Parsing
# -------------------------------
parser = ArgumentParser(description="SpectralNet GNN")
parser.add_argument('--raw_dir', type=str, default='spectral_paths', help="Directory where the data is stored")
parser.add_argument('--hidden_dim', type=int, default=250, help="Hidden dim for spectral MLP")
parser.add_argument('--num_eigenvectors', type=int, default=100, help="Number of eigenvectors")
parser.add_argument('--lr', type=float, default=1e-2, help="Learning Rate")
parser.add_argument('--wd', type=float, default=1e-2, help="Weight decay")
parser.add_argument('--num_epochs', type=int, default=300, help="Number of epochs")
parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
parser.add_argument('--gpu', type=int, default=0, help="GPU index")
parser.add_argument('--num_runs', type=int, default=10, help="Number of runs for repeated training")
parser.add_argument('--dataset', type=str, default='pems07', choices=['abide', 'pvdm', 'ra', 'toy', 'toy_er', 'pems03', 'pems07'], help="Dataset to use")
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
            out = model(batch.x, batch.edge_index, batch.batch)
            preds = torch.argmax(out, dim=1)
            correct += torch.sum(preds == batch.y).float()
            total += batch.y.size(0)
    return (correct * 100) / total

# -------------------------------
# Training Function
# -------------------------------
def train(model, run_id):
    print(f"Starting Run {run_id + 1}/{args.num_runs}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    best_acc = 0

    with tqdm(range(args.num_epochs)) as tq:
        for epoch in tq:
            model.train()
            correct_train = 0
            t_loss = 0

            for batch in train_loader:
                batch = batch.to(args.device)
                opt.zero_grad()
                out = model(batch.x, batch.edge_index, batch.batch)
                preds = torch.argmax(out, dim=1)
                # loss = loss_fn(out, batch.y)
                loss = loss_fn(out, batch.y.squeeze(-1).long())
                loss.backward()
                opt.step()

                correct_train += torch.sum(preds == batch.y).float()
                t_loss += loss.item()

            train_acc = (correct_train * 100) / len(train_graphs)
            test_acc = test(model, test_loader)

            wandb.log({'Run': run_id, 'Loss': t_loss, 'Train acc': train_acc, 'Test acc': test_acc}, step=epoch+1)

            if test_acc > best_acc:
                best_acc = test_acc
                model_path = f"{args.raw_dir}/spectralnet_best_run_{run_id}_{args.dataset}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'best_acc': best_acc,
                    'args': args
                }, model_path)

            tq.set_description(f"[Run {run_id+1}] Train acc = {train_acc:.2f}, Test acc = {test_acc:.2f}, Best acc = {best_acc:.2f}")

    print(f"Best accuracy for Run {run_id + 1}: {best_acc}")
    return best_acc

# -------------------------------
# Main
# -------------------------------
if __name__ == '__main__':
    print("Loading graph dataset...")
    if args.dataset == 'abide':
        print("Loading graph dataset...")
        graph_dataset = torch.load("BioPoint/abide_graphs_balanced.pt")  # List of PyG Data objects
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

    elif args.dataset == 'pems03':
        graph_dataset = torch.load("traffic_data/PEMS03/graphs.pt")
        adj = graph_dataset[0].edge_index
        adj = to_dense_adj(adj)[0]
        clusters_df = pd.read_csv("traffic_data/PEMS03/pems03_clusters.csv")
        # Convert the clusters to a dictionary
        clusters_dict = {col: clusters_df[col].dropna().tolist() for col in clusters_df.columns}
        clusters_dict = {k: [int(v) for v in vals] for k, vals in clusters_dict.items()}
        cluster_names = list(clusters_dict.keys())
        num_clusters = len(cluster_names)
        # label_counts = {0: 0, 1: 0}
        # for graph in graph_dataset:
        #     label_counts[int(graph.y.item())] += 1

        # print(f"Number of graphs with label 0: {label_counts[0]}")
        # print(f"Number of graphs with label 1: {label_counts[1]}")
        # import pdb; pdb.set_trace()
        num_nodes = graph_dataset[0].x.shape[0]  # Number of nodes in the first graph  
        # import pdb; pdb.set_trace()
        cluster_masks = torch.zeros(num_clusters, num_nodes)
        for idx, cluster in enumerate(cluster_names):
            for node in clusters_dict[cluster]:
                cluster_masks[idx, node] = 1.0
    elif args.dataset == 'pems07':
        graph_dataset = torch.load("traffic_data/PEMS07/graphs.pt")
        adj = graph_dataset[0].edge_index
        adj = to_dense_adj(adj)[0]
        clusters_df = pd.read_csv("traffic_data/PEMS07/pems07_clusters.csv")
        # Convert the clusters to a dictionary
        clusters_dict = {col: clusters_df[col].dropna().tolist() for col in clusters_df.columns}
        clusters_dict = {k: [int(v) for v in vals] for k, vals in clusters_dict.items()}
        cluster_names = list(clusters_dict.keys())
        num_clusters = len(cluster_names)
        # label_counts = {0: 0, 1: 0}
        # for graph in graph_dataset:
        #     label_counts[int(graph.y.item())] += 1

        # print(f"Number of graphs with label 0: {label_counts[0]}")
        # print(f"Number of graphs with label 1: {label_counts[1]}")
        # import pdb; pdb.set_trace()
        num_nodes = graph_dataset[0].x.shape[0]  # Number of nodes in the first graph  
        # import pdb; pdb.set_trace()
        cluster_masks = torch.zeros(num_clusters, num_nodes)
        for idx, cluster in enumerate(cluster_names):
            for node in clusters_dict[cluster]:
                cluster_masks[idx, node] = 1.0
    
    
    all_best_accs = []

    for run_id in range(args.num_runs):
        train_idx, test_idx = train_test_split(np.arange(len(graph_dataset)), test_size=0.2, random_state=run_id)
        train_graphs = [graph_dataset[i] for i in train_idx]
        test_graphs = [graph_dataset[i] for i in test_idx]

        train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_graphs, batch_size=args.batch_size)

        print(f"Dataset Loaded: {len(train_graphs)} training graphs, {len(test_graphs)} test graphs")

        wandb.init(project='spectralnet-training', name=f"spectral_run_{run_id}", config=vars(args), reinit=True)

        if args.gpu != -1 and torch.cuda.is_available():
            args.device = f'cuda:{args.gpu}'
        else:
            args.device = 'cpu'

        num_nodes = 42
        num_features = train_graphs[0].x.size(1)
        num_classes = 7

        model = SpectralNet(
            num_nodes=num_nodes,
            num_features=num_features,
            num_classes=num_classes,
            hidden_dim=args.hidden_dim,
            num_eigenvectors=args.num_eigenvectors,
            adj=adj
        ).to(args.device)

        best_acc = train(model, run_id)
        all_best_accs.append(best_acc)

    mean_acc = np.mean([acc.cpu().item() for acc in all_best_accs])
    std_acc = np.std([acc.cpu().item() for acc in all_best_accs])

    print(f"\nFinal Summary Over {args.num_runs} Runs:")
    print(f"Mean Best Accuracy: {mean_acc:.2f}%")
    print(f"Standard Deviation: {std_acc:.2f}%")

    wandb.init(project='spectralnet-training', name="spectralnet-summary", config=vars(args), reinit=True)
    wandb.log({'Mean Best Accuracy': mean_acc, 'Standard Deviation': std_acc})
