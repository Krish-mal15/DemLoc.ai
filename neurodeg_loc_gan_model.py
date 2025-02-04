import torch.nn as nn
import torch
from torch_geometric.nn import GAT, GIN

# This will be a conditional GAN (cGAN). The model will arbitrarily score EEG signals based on graph isomorphic structures.
# The generator loss will be based off of discriminator scoring of the predicted matrix by using graph attention. The prediction
# will also be influenced by embedded MMSE scores

# Input data is PSD as nodes and the interregional connectivity of the EEG signal, not PSD data.

class ScoringConnectivityGenerator(nn.Module):
    def __init__(self, n_psd_freqs):
        super().__init__()
        
        self.gat = GAT(
            in_channels=n_psd_freqs,
            hidden_channels=512,
            out_channels=256,
            num_layers=4
        )
        
        self.linear_scorer = nn.Linear(19, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, eeg_graph):
        gat_out = self.gat(eeg_graph.x, eeg_graph.edge_index, eeg_graph.edge_attr)
        conn_reconstructed = torch.matmul(gat_out, gat_out.T)
        
        region_scores = self.sigmoid(self.linear_scorer(conn_reconstructed))
        return region_scores
    
class DementiaConditioningDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.gin_discrim_latent = GIN(
            in_channels=19,
            hidden_channels=128,
            out_channels=64,
            num_layers=4
        )
        
        # Using MMSE projeciton conditioning will help model learn GIN features that correlate to MMSE in discrimination as well
        self.mmse_predictor = nn.Sequential(
            nn.Linear(64, 1),
            nn.LeakyReLU()
        )
        
        self.gin_main_discrim = GIN(
            in_channels=64,
            hidden_channels=32,
            out_channels=1,
            num_layers=1
        )
        
    def forward(self, regional_scores_graph_struct):
        discrim_latent_out = self.gin_discrim_latent(regional_scores_graph_struct.x, regional_scores_graph_struct.edge_index)
        mmse_pred = self.mmse_predictor(discrim_latent_out)
        
        discrim_out = self.gin_main_discrim(discrim_latent_out, regional_scores_graph_struct.edge_index)
        return discrim_out, mmse_pred
