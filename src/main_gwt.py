import torch
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj
import wandb
import pandas as pd
import argparse
import gc

from src.GWT import GraphWaveletTransform  # you can rename if needed

gc.enable()

class MLPClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=2):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def test(model, gwt, loader, loss_fn, device):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            gwt.X_init = batch.x
            gwt.edge_index = batch.edge_index
            gwt.edge_weight = torch.ones(batch.edge_index.size(1)).to(device)

            feat = gwt.generate_timepoint_features(batch.batch)
            out = model(feat)
            # loss = loss_fn(out, batch.y)
            loss = loss_fn(out, batch.y.squeeze(-1).long())
            preds = torch.argmax(out, dim=1)

            test_loss += loss.item()
            correct += (preds == batch.y).sum().float()
            total += batch.y.size(0)

    return (correct * 100) / total, test_loss


def train(model, gwt, train_loader, test_loader, device, num_epochs, run_id):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = 0
    for epoch in tqdm(range(num_epochs)):
        model.train()
        correct_train = 0
        total_loss = 0

        for batch in train_loader:
            batch = batch.to(device)
            gwt.X_init = batch.x
            gwt.edge_index = batch.edge_index
            gwt.edge_weight = torch.ones(batch.edge_index.size(1)).to(device)

            feat = gwt.generate_timepoint_features(batch.batch)
            out = model(feat)
            # loss = loss_fn(out, batch.y)
            loss = loss_fn(out, batch.y.squeeze(-1).long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(out, dim=1)
            correct_train += (preds == batch.y).sum().float()

        train_acc = (correct_train * 100) / len(train_loader.dataset)
        test_acc, test_loss = test(model, gwt, test_loader, loss_fn, device)

        wandb.log({'Run': run_id, 'Train Loss': total_loss, 'Test Loss': test_loss, 'Train acc': train_acc, 'Test acc': test_acc}, step=epoch+1)

        if test_acc > best_acc:
            best_acc = test_acc
            model_path = f"{args.raw_dir}/GWT_best_run_{run_id}_{args.dataset}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'args': args
            }, model_path)

        # tqdm.write(f"[Run {run_id+1}] Train acc = {train_acc:.2f}, Test acc = {test_acc:.2f}, Best acc = {best_acc:.2f}")
    print(f"Best accuracy for Run {run_id + 1}: {best_acc}")
    return best_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', type=str, default='GWT_Paths', help="Directory where the data is stored")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_runs', type=int, default=10)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--J', type=int, default=3)  # Wavelet depth
    parser.add_argument('--dataset', type=str, default='pems07', choices=['abide', 'pvdm', 'ra', 'toy', 'toy_er', 'pems03', 'pems07'], help="Dataset to use")
    args = parser.parse_args()

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu != -1 else 'cpu'

    if args.dataset == 'abide':
        graph_dataset = torch.load("BioPoint/abide_graphs_balanced.pt")
    elif args.dataset == 'pvdm':
        graph_dataset = torch.load("OCD_data/all_graphs_time_emb.pt")
    elif args.dataset == 'ra':
        graph_dataset = torch.load("OCD_2_DATA/all_graphs_time_emb_RA.pt")
    elif args.dataset == 'pems03':
        graph_dataset = torch.load("traffic_data/PEMS03/graphs.pt")
    elif args.dataset == 'pems07':
        graph_dataset = torch.load("traffic_data/PEMS07/graphs.pt")

    print(f"Loaded dataset: {len(graph_dataset)} graphs")

    dummy = graph_dataset[0]
    input_dim = dummy.x.shape[1]
    num_classes = 7

    all_accs = []
    for run_id in range(args.num_runs):
        train_idx, test_idx = train_test_split(np.arange(len(graph_dataset)), test_size=0.2, random_state=run_id)
        train_graphs = [graph_dataset[i] for i in train_idx]
        test_graphs = [graph_dataset[i] for i in test_idx]

        train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_graphs, batch_size=args.batch_size)

        wandb.init(project='gwt-training', name=f"gwt_run_{run_id}", config=vars(args), reinit=True)

        gwt = GraphWaveletTransform(
            edge_index=dummy.edge_index,
            edge_weight=torch.ones(dummy.edge_index.size(1)),
            X=dummy.x,
            J=args.J,
            device=device
        )

        sample_feat = gwt.generate_timepoint_features(torch.zeros(dummy.x.size(0), dtype=torch.long, device=device))
        model = MLPClassifier(sample_feat.size(1), num_classes=num_classes).to(device)

        best_acc = train(model, gwt, train_loader, test_loader, device, args.num_epochs, run_id)
        all_accs.append(best_acc)

    mean_acc = np.mean([acc.item() for acc in all_accs])
    std_acc = np.std([acc.item() for acc in all_accs])

    print(f"\nFinal Summary Over {args.num_runs} Runs:")
    print(f"Mean Best Accuracy: {mean_acc:.2f}%")
    print(f"Standard Deviation: {std_acc:.2f}%")

    wandb.init(project='gwt-training', name="summary", config=vars(args), reinit=True)
    wandb.log({'Mean Best Accuracy': mean_acc, 'Standard Deviation': std_acc})
