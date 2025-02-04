import torch.nn as nn
from torch_geometric.nn import GIN
import torch

from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data

from eeg_graph_construction import eeg_sim_matrix_calc
from data_processing_loading import get_single_sample_eeg

import umap
import matplotlib.pyplot as plt

# From experimental results, correlating EEG with phase locking value (PLV) led to worse results so
# all graph operations should only take in nodes and edge_index (still use PLV-derived adj_matrix)

def latent_test_procedure(latent_vector_z):
    reducer = umap.UMAP(n_neighbors=10, min_dist=0.1, random_state=42)
    latent_2d = reducer.fit_transform(latent_vector_z)

    # Plot the 2D representation
    plt.figure(figsize=(8, 6))
    plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c='blue', edgecolor='k', s=100)
    plt.title("UMAP Visualization of Latent Space")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True)
    plt.show()
    

def reparameterize(mean, var):
    epsilon = torch.randn_like(var).to(torch.device('cpu'))
    z = mean + var * epsilon
    return z
    
class DemLocGraphEncoder(nn.Module):
    def __init__(self, n_eeg_timesteps, latent_dim):
        super().__init__()
        
        self.gin_encoder = GIN(
            in_channels=n_eeg_timesteps,
            hidden_channels=2048,
            out_channels=1024,
            num_layers=4
        )
        
        # Try getting latent vector to (19, latent_dim) so node-wise PCA could maybe work
        self.latent_mean_transform = nn.Linear(1024, latent_dim)
        self.latent_var_transform = nn.Linear(1024, latent_dim)
        
    def forward(self, eeg_nodes, eeg_idx):
        graph_encoded_features = self.gin_encoder(eeg_nodes, eeg_idx)
        
        mean = self.latent_mean_transform(graph_encoded_features)
        var = self.latent_var_transform(graph_encoded_features)
        
        z = reparameterize(mean, var)
        
        return z, mean, var

class DemLocDecoder(nn.Module):
    def __init__(self, n_eeg_timesteps, latent_dim):
        super().__init__()
        
        self.gin_recon = GIN(
            in_channels=latent_dim,
            hidden_channels=2048,
            out_channels=n_eeg_timesteps,
            num_layers=2
        )
        
        # self.reg_classifier = nn.Sequential(
        #     nn.Linear(n_eeg_timesteps, latent_dim),
        #     nn.Linear(latent_dim, 1),
        #     nn.Sigmoid()
        # )
        
        self.dem_classifier = nn.Sequential(
            nn.Linear(n_eeg_timesteps * 19, latent_dim),
            nn.Linear(latent_dim, 1),
            nn.Sigmoid()
        )
        
        
    def forward(self, latent_z, edge_idx):
        gin_recon_signal = self.gin_recon(latent_z, edge_idx)

        # reg_pred = self.reg_classifier(gin_recon_signal)
        gin_recon_signal_flat = gin_recon_signal.reshape(19 * 4096)
        dem_pred = self.dem_classifier(gin_recon_signal_flat)
        
        return dem_pred, gin_recon_signal
    
class DemLocVAE(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = DemLocGraphEncoder(n_eeg_timesteps=4096, latent_dim=256)
        self.decoder = DemLocDecoder(n_eeg_timesteps=4096, latent_dim=256)
        
    def forward(self, eeg_nodes, eeg_idx):
        encoded_z, mu, log_var = self.encoder(eeg_nodes, eeg_idx)
        decoded_dem_pred, recon_signal = self.decoder(encoded_z, eeg_idx)
        
        return decoded_dem_pred, recon_signal, mu, log_var, encoded_z

# Could be trained separately with the VAE returning encoded vector z (with useful trained info) and then this model can give
# regional scores
class RegionalLatentClassifier(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        
        self.reg_class = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.Linear(64, 19)
        )
        
    def forward(self, encoded_z):
        reg_scores = self.reg_class(encoded_z)
        return reg_scores
            

# model = DemLocVAE()

# test_eeg_signal = torch.tensor(get_single_sample_eeg('EEGNeurodegenerationDatasetClassed/Dementia/sub-003/eeg/sub-003_task-eyesclosed_eeg.set'), dtype=torch.float)
# test_adj = torch.tensor(eeg_sim_matrix_calc(test_eeg_signal, sfreq=500), dtype=torch.float)
# test_eeg_index, test_eeg_attr = dense_to_sparse(test_adj)

# test_eeg_index = torch.tensor(test_eeg_index, dtype=torch.int64)
# test_eeg_attr = torch.tensor(test_eeg_attr, dtype=torch.float)

# # print("Nodes: ", test_eeg_signal)
# # print("Index: ", test_eeg_index)

# test_eeg_graph = Data(x=test_eeg_signal, edge_index=test_eeg_index, edge_attr=test_eeg_attr)
# print("EEG Test Graph: ", test_eeg_graph)

# model_out, encoded_z = model(test_eeg_graph.x, test_eeg_graph.edge_index)
# print(model_out)

# encoded_z = encoded_z.detach().numpy()

# latent_test_procedure(encoded_z)


