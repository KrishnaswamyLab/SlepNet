import torch
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
import wandb
import gc
from argparse import ArgumentParser
from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv, global_mean_pool, global_add_pool
import torch.nn.functional as F
import torch.nn as nn
import pickle

gc.enable()

# -------------------------------
# Argument Parsing for Training
# -------------------------------
parser = ArgumentParser(description="Graph Neural Network Classifier")
parser.add_argument('--raw_dir', type=str, default='gnn_paths_PEMS', help="Directory where the data is stored")
parser.add_argument('--task', type=str, default='classification', help="Classification task type")
parser.add_argument('--hidden_dims', type=int, nargs='+', default=[128,64], help="Hidden dimensions per layer")
parser.add_argument('--lr', type=float, default=1e-3, help="Learning Rate")
parser.add_argument('--wd', type=float, default=1e-2, help="Weight decay")
parser.add_argument('--num_epochs', type=int, default=100, help="Number of epochs")
parser.add_argument('--batch_size', type=int, default=128, help="Batch size")
parser.add_argument('--gpu', type=int, default=0, help="GPU index")
parser.add_argument('--model', type=str, default='GCN', choices=['GCN', 'GIN', 'GAT', 'GraphSAGE', 'TransformerGNN'], help="GNN model type")
parser.add_argument('--num_runs', type=int, default=10, help="Number of training runs")
parser.add_argument('--device', type=str, default='cuda', help="Device to run the model on")
parser.add_argument('--dataset', type=str, default='pems07', choices=['abide', 'pvdm', 'ra', 'toy', 'toy_er', 'pems03', 'pems07'], help="Dataset to use")
args = parser.parse_args()

# -------------------------------
# Define GNN Models
# -------------------------------
class GCN(nn.Module):
    def __init__(self, num_features, hidden_dims, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dims[0])
        self.conv2 = GCNConv(hidden_dims[0], hidden_dims[1])
        self.conv3 = GCNConv(hidden_dims[1],32)
        self.conv4 = GCNConv(32, 16)
        self.linear = nn.Linear(16, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = F.relu(self.conv4(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.linear(x)
    
    def encode(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        # x = F.relu(self.conv3(x, edge_index))
        # x = F.relu(self.conv4(x, edge_index))
        return x

class GIN(nn.Module):
    def __init__(self, num_features, hidden_dims, num_classes):
        super(GIN, self).__init__()
        self.conv1 = GINConv(nn.Linear(num_features, hidden_dims[0]))
        self.conv2 = GINConv(nn.Linear(hidden_dims[0], hidden_dims[1]))
        self.linear = nn.Linear(hidden_dims[1], num_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.linear(x)
    
    def encode(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return x
    


class GAT(nn.Module):
    def __init__(self, num_features, hidden_dims, num_classes):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_features, hidden_dims[0])
        self.conv2 = GATConv(hidden_dims[0], hidden_dims[1])
        self.linear = nn.Linear(hidden_dims[1], num_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.linear(x)
    
    def encode(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return x

class GraphSAGE(nn.Module):
    def __init__(self, num_features, hidden_dims, num_classes):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(num_features, hidden_dims[0])
        self.conv2 = SAGEConv(hidden_dims[0], hidden_dims[1])
        self.linear = nn.Linear(hidden_dims[1], num_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.linear(x)

    def encode(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return x

class TransformerGNN(nn.Module):
    def __init__(self, num_features, hidden_dims, num_classes, num_heads=4, num_layers=2, dropout=0.1):
        super(TransformerGNN, self).__init__()
        # self.embedding = nn.Linear(num_features, hidden_dims[0])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=num_features, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(hidden_dims[0], num_classes)

    def forward(self, x, edge_index, batch):
        x = self.embedding(x)
        batch_size = batch.max().item() + 1
        node_counts = torch.bincount(batch)
        max_nodes = node_counts.max().item()

        padded_x = torch.zeros(batch_size, max_nodes, x.size(1), device=x.device)
        mask = torch.ones(batch_size, max_nodes, device=x.device, dtype=torch.bool)

        start_idx = 0
        for i, count in enumerate(node_counts):
            padded_x[i, :count] = x[start_idx : start_idx + count]
            mask[i, :count] = False
            start_idx += count

        x = self.transformer(padded_x, src_key_padding_mask=mask)
        x = x.mean(dim=1)
        return self.linear(x)

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
                # loss = loss_fn(out, batch.y)
                loss = loss_fn(out, batch.y.squeeze(-1).long())
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
                model_path = f"{args.raw_dir}/{args.model}_best_run_{run_id}_{args.dataset}.pth"
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

# -------------------------------
# Main Execution
# -------------------------------
if __name__ == '__main__':
    print("Loading graph dataset...")
    if args.dataset == 'pems07':
        graph_dataset = torch.load("traffic_data/PEMS07/graphs.pt")  # List of PyG Data objects
    elif args.dataset == 'pems03':
        graph_dataset = torch.load("traffic_data/PEMS03/graphs.pt")
    # with open(f"{args.raw_dir}/fnirs_graphs_deoxy.pkl", 'rb') as f:
    #     graph_dataset = pickle.load(f)
    # import pdb; pdb.set_trace()
    # for graph in graph_dataset:
    #     graph.x = (graph.x - graph.x.mean(dim=0)) / (graph.x.std(dim=0) + 1e-6)
    all_best_accs = []

    for run_id in range(args.num_runs):
        train_idx, test_idx = train_test_split(np.arange(len(graph_dataset)), test_size=0.2, random_state=run_id)
        train_graphs = [graph_dataset[i] for i in train_idx]
        test_graphs = [graph_dataset[i] for i in test_idx]

        train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_graphs, batch_size=args.batch_size)

        wandb.init(project=f'{args.model}-training', name=f"run_{run_id}", config=vars(args), reinit=True)

        model_classes = {'GCN': GCN, 'GIN': GIN, 'GAT': GAT, 'GraphSAGE': GraphSAGE, 'TransformerGNN': TransformerGNN}
        model = model_classes[args.model](num_features=1, hidden_dims=args.hidden_dims, num_classes=7).to(args.device)

        best_acc = train(model, run_id)
        all_best_accs.append(best_acc)
        mean_acc = np.mean([acc.cpu().item() for acc in all_best_accs])
        std_acc = np.std([acc.cpu().item() for acc in all_best_accs])
    print(f"Mean Best Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
