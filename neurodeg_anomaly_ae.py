import torch.nn as nn
import torch

from torch_geometric.nn import GIN
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data

from eeg_graph_construction import eeg_sim_matrix_calc
from data_processing_loading import get_single_sample_eeg

device = torch.device('cpu')

# Latent dimension of the autoencoder should be larger since its raw features and not an expressive distribution.
class GraphEncoder(nn.Module):
    def __init__(self, n_eeg_timesteps, latent_dim):
        super().__init__()
        
        self.gin_encoder = GIN(
            in_channels=n_eeg_timesteps,
            hidden_channels=n_eeg_timesteps//2,
            out_channels=latent_dim * 2,
            num_layers=2
        )
        
        self.latent_transform = nn.Linear(latent_dim * 2, latent_dim)
        
    def forward(self, eeg_graph):
        graph_features = self.gin_encoder(eeg_graph.x, eeg_graph.edge_index, eeg_graph.edge_attr)
        # print(graph_features)
        latent_features = self.latent_transform(graph_features)
        
        return latent_features # (19, latent_dim)
    
class GraphDecoder(nn.Module):
    def __init__(self, n_eeg_timesteps, latent_dim, recon_method):
        super().__init__()
        self.recon_method = recon_method
        
        self.inverse_linear_transform = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(latent_dim * 2, n_eeg_timesteps)
        )
        
    def forward(self, latent_features):
        # 'adj' reconstruction methods reconstructs the functional connectivity (PLV adj), while other options just
        # reconstruct the signal through learnable linear transformations
        if self.recon_method == 'adj':
            decoded_out = torch.matmul(latent_features, latent_features.T)
        else:
            decoded_out = self.inverse_linear_transform(latent_features)
        return decoded_out
    
class GraphAnomalyAE(nn.Module):
    def __init__(self, device):
        super().__init__()
        
        self.encoder = GraphEncoder(n_eeg_timesteps=4096, latent_dim=512).to(device)
        self.decoder = GraphDecoder(n_eeg_timesteps=4096, latent_dim=512, recon_method='sig').to(device)
       
    def forward(self, eeg_graph):
        encoded = self.encoder(eeg_graph)
        decoded = self.decoder(encoded)
        
        return decoded, encoded

model = GraphAnomalyAE(device)

# # # --- Model Arbitrary Testing --- #
# test_eeg_signal = torch.tensor(get_single_sample_eeg('EEGNeurodegenerationDatasetClassed/Dementia/sub-003/eeg/sub-003_task-eyesclosed_eeg.set'), dtype=torch.float)
# test_adj = torch.tensor(eeg_sim_matrix_calc(test_eeg_signal, sfreq=500), dtype=torch.float)
# test_eeg_index, test_eeg_attr = dense_to_sparse(test_adj)

# test_eeg_index = torch.tensor(test_eeg_index, dtype=torch.int64)
# test_eeg_attr = torch.tensor(test_eeg_attr, dtype=torch.float)

# test_eeg_graph = Data(x=test_eeg_signal, edge_index=test_eeg_index, edge_attr=test_eeg_attr)
# # print("EEG Graph: ", test_eeg_graph)

# decoded, latent_features = model(test_eeg_graph)
# # print(decoded)
        
