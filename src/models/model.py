
import time
import numpy as np
from scipy.spatial.distance import cdist
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_graph_laplacian
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from numpy import linalg as LA
from pygsp import graphs, plotting
# import matplotlib.pyplot as plt

class SpectralGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SpectralGNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Learnable spectral filters
        self.filter1 = nn.Parameter(torch.randn(input_dim, hidden_dim))
        self.filter2 = nn.Parameter(torch.randn(hidden_dim, output_dim))
        
    # def compute_laplacian(self, edge_index, num_nodes):
    #     # Compute adjacency matrix
    #     adj = torch.zeros(num_nodes, num_nodes)
    #     adj[edge_index[0], edge_index[1]] = 1
        
    #     # Compute degree matrix
    #     degree = torch.diag(adj.sum(dim=1))
        
    #     # Compute normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
    #     degree_inv_sqrt = torch.inverse(torch.sqrt(degree))
    #     laplacian = torch.eye(num_nodes) - degree_inv_sqrt @ adj @ degree_inv_sqrt
        
    #     return laplacian
    
    def eigendecomposition(self, laplacian):
        # Compute eigenvalues and eigenvectors using scipy (for CPU)
        laplacian_np = laplacian.numpy()
        eigenvalues, eigenvectors = eigh(laplacian_np)
        
        # Convert back to PyTorch tensors
        eigenvalues = torch.from_numpy(eigenvalues).float()
        eigenvectors = torch.from_numpy(eigenvectors).float()
        
        return eigenvalues, eigenvectors
    
    def forward(self, x, laplacian):
        num_nodes = x.size(0)
        
        # Eigendecomposition of Laplacian
        eigenvalues, eigenvectors = self.eigendecomposition(laplacian)
        
        # Transform input signal to spectral domain: U^T x
        x_spectral = eigenvectors.t() @ x
        
        # Apply first spectral filter
        x_spectral = x_spectral @ self.filter1
        x_spectral = F.relu(x_spectral)  # Apply nonlinearity
        
        # Apply second spectral filter
        x_spectral = x_spectral @ self.filter2
        
        # Transform back to spatial domain: U x_spectral
        x_spatial = eigenvectors @ x_spectral
        
        return x_spatial  # Output for regression (no softmax)

# Generate a grid graph using PyGSP
# def generate_grid_graph(grid_size):
#     # Create a 2D grid graph using PyGSP
#     G = graphs.Grid2d(N1=grid_size, N2=grid_size)
#     G.compute_laplacian('normalized')  # Compute normalized Laplacian
#     edge_list = G.get_edge_list()
#     edge_list = np.asarray(edge_list)
#     edge_list = edge_list[[0,1], :]
#     print(edge_list)

    
#     # Get edge_index in PyTorch format
#     edge_index = torch.tensor(edge_list,  dtype=torch.uint8).t().contiguous()
    
#     return G, edge_index

# Plot the grid graph using PyGSP
# def plot_grid_graph(G, node_features=None):
#     plt.figure(figsize=(6, 6))
#     G.set_coordinates('line2D')  # Set node positions for visualization
#     plotting.plot_graph(G, show_edges=True, vertex_size=200)
    
#     if node_features is not None:
#         for i in range(len(node_features)):
#             plt.text(G.coords[i, 0], G.coords[i, 1], f"{i}\n{node_features[i].argmax().item()}", 
#                      fontsize=8, ha='center', va='center')
    
#     plt.title("Grid Graph")
#     plt.show()

class SlepNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_nodes):
        super(SpectralGNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Learnable spectral filters
        self.filter1 = nn.Parameter(torch.randn(input_dim, hidden_dim))
        self.filter2 = nn.Parameter(torch.randn(hidden_dim, output_dim))

        #Node selection Mask
        self.selection = nn.Parameter(torch.randn(num_nodes))
        
    # def compute_laplacian(self, edge_index, num_nodes):
    #     # Compute adjacency matrix
    #     adj = torch.zeros(num_nodes, num_nodes)
    #     adj[edge_index[0], edge_index[1]] = 1
        
    #     # Compute degree matrix
    #     degree = torch.diag(adj.sum(dim=1))
        
    #     # Compute normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
    #     degree_inv_sqrt = torch.inverse(torch.sqrt(degree))
    #     laplacian = torch.eye(num_nodes) - degree_inv_sqrt @ adj @ degree_inv_sqrt
        
    #     return laplacian
    
    def eigendecomposition(self, laplacian):
        # Compute eigenvalues and eigenvectors using scipy (for CPU)
        laplacian_np = laplacian.numpy()
        eigenvalues, eigenvectors = eigh(laplacian_np)
        
        # Convert back to PyTorch tensors
        eigenvalues = torch.from_numpy(eigenvalues).float()
        eigenvectors = torch.from_numpy(eigenvectors).float()
        
        return eigenvalues, eigenvectors
    
    def forward(self, x, laplacian, S):
        num_nodes = x.size(0)
        
        # Eigendecomposition of Laplacian
        eigenvalues, eigenvectors = self.eigendecomposition(laplacian)
        
        # Transform input signal to spectral domain: U^T x
        x_spectral = eigenvectors.t() @ x
        
        # Apply first spectral filter
        x_spectral = x_spectral @ self.filter1
        x_spectral = F.relu(x_spectral)  # Apply nonlinearity
        
        # Apply second spectral filter
        x_spectral = x_spectral @ self.filter2
        
        # Transform back to spatial domain: U x_spectral
        x_spatial = eigenvectors @ x_spectral
        
        return x_spatial  # Output for regression (no softmax)


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.linalg import eigh
from pygsp import graphs, plotting
import matplotlib.pyplot as plt

class SlepNet_Masked_GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,a num_nodes):
        super(SpectralGNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_nodes = num_nodes
        
        # Learnable spectral filters
        self.filter1 = nn.Parameter(torch.randn(input_dim, hidden_dim))
        self.filter2 = nn.Parameter(torch.randn(hidden_dim, output_dim))
        
        # Learnable mask (initialized randomly)
        self.mask_logits = nn.Parameter(torch.randn(num_nodes))
        
        # Precompute Slepian basis (will be updated during training)
        self.slepian_basis = None

    def compute_laplacian(self, edge_index):
        # Compute adjacency matrix
        adj = torch.zeros(self.num_nodes, self.num_nodes)
        adj[edge_index[0], edge_index[1]] = 1
        
        # Compute degree matrix
        degree = torch.diag(adj.sum(dim=1))
        
        # Compute normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
        degree_inv_sqrt = torch.inverse(torch.sqrt(degree))
        laplacian = torch.eye(self.num_nodes) - degree_inv_sqrt @ adj @ degree_inv_sqrt
        
        return laplacian
    
    def compute_slepian_basis(self, mask):
        # Compute the Slepian basis for the selected nodes
        G = graphs.Grid2d(N1=int(np.sqrt(self.num_nodes)), N2=int(np.sqrt(self.num_nodes)))
        G.compute_laplacian('normalized')
        
        # Focus on the subgraph defined by the mask
        subgraph_indices = torch.where(mask > 0.5)[0]  # Use thresholded mask
        subgraph_laplacian = G.L[subgraph_indices][:, subgraph_indices]
        
        # Eigendecomposition of the subgraph Laplacian
        eigenvalues, eigenvectors = eigh(subgraph_laplacian.toarray())
        slepian_basis = torch.from_numpy(eigenvectors).float()
        
        # Map back to the original graph
        full_basis = torch.zeros(self.num_nodes, slepian_basis.size(1))
        full_basis[subgraph_indices] = slepian_basis
        
        return full_basis
    
    def forward(self, x, edge_index):
        # Compute the mask using a sigmoid activation
        mask = torch.sigmoid(self.mask_logits)  # Soft mask (values between 0 and 1)
        
        # Update the Slepian basis based on the current mask
        self.slepian_basis = self.compute_slepian_basis(mask)
        
        # Compute Laplacian
        laplacian = self.compute_laplacian(edge_index)
        
        # Transform input signal to Slepian domain: S^T x
        x_slepian = self.slepian_basis.t() @ x
        
        # Apply first spectral filter in Slepian domain
        x_slepian = x_slepian @ self.filter1
        x_slepian = F.relu(x_slepian)  # Apply nonlinearity
        
        # Apply second spectral filter in Slepian domain
        x_slepian = x_slepian @ self.filter2
        
        # Transform back to spatial domain: S x_slepian
        x_spatial = self.slepian_basis @ x_slepian
        
        return x_spatial, mask  # Output for regression (no softmax) and the learned mask

# Generate a grid graph using PyGSP
def generate_grid_graph(grid_size):
    # Create a 2D grid graph using PyGSP
    G = graphs.Grid2d(N1=grid_size, N2=grid_size)
    G.compute_laplacian('normalized')  # Compute normalized Laplacian
    
    # Get edge_index in PyTorch format
    edge_index = torch.tensor(G.get_edge_list()).t().contiguous()
    
    return G, edge_index

# Plot the grid graph using PyGSP
def plot_grid_graph(G, node_features=None, mask=None):
    plt.figure(figsize=(6, 6))
    G.set_coordinates('grid2d')  # Set node positions for visualization
    
    # Highlight nodes based on the mask
    if mask is not None:
        node_colors = ['red' if mask[i] > 0.5 else 'lightblue' for i in range(len(mask))]
        plotting.plot_graph(G, show_edges=True, vertex_size=200, vertex_color=node_colors)
    else:
        plotting.plot_graph(G, show_edges=True, vertex_size=200, vertex_color='lightblue')
    
    if node_features is not None:
        for i in range(len(node_features)):
            plt.text(G.coords[i, 0], G.coords[i, 1], f"{i}\n{node_features[i].argmax().item()}", 
                     fontsize=8, ha='center', va='center')
    
    plt.title("Grid Graph with Learnable Node Selection Mask")
    plt.show()

if __name__ == "__main__":
    # Generate a 4x4 grid graph
    grid_size = 4
    G, edge_index = generate_grid_graph(grid_size)
    
    # Initialize node features as one-hot vectors
    num_nodes = G.N
    x = torch.eye(num_nodes)  # One-hot encoding
    
    # Generate random target values for regression (output_dim = 1)
    y = torch.randn(num_nodes, 1, dtype=torch.float)
    
    # Initialize the model
    model = SpectralGNN(input_dim=num_nodes, hidden_dim=16, output_dim=1, num_nodes=num_nodes)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        output, mask = model(x, edge_index)
        
        # Compute loss
        loss = criterion(output, y)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    
    # Test the model
    model.eval()
    with torch.no_grad():
        predicted, mask = model(x, edge_index)
        print("Predicted values:", predicted.squeeze().numpy())
        print("True values:", y.squeeze().numpy())
        print("Learned mask:", (mask > 0.5).int().numpy())  # Thresholded mask
    
    # Plot the grid graph with the learned mask
    plot_grid_graph(G, node_features=x, mask=mask.detach().numpy())
