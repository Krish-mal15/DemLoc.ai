# Similar architecture to V1, but this version allows you to input the MMSE instead of just learning those features in training.
# WIll use only graph isomorphism with graph attention as a "bottleneck" to apply attention to learned structure features.
# Will use MMSE inverse to condition graph isomorphic scores by enhancing already high scored regions if MMSE is low,
# while decreasing scores is MMSE is high
# This approach can also help MSE loss from skewing binary classification thats more important for total propagation

import torch.nn as nn
from torch_geometric.nn import GAT, GIN
from torch_geometric.utils import dense_to_sparse

import torch
import math
import numpy as np

def compute_neural_score_adj(regional_scores):
    num_nodes = regional_scores.shape[0]
    adjacency_matrix = np.zeros((num_nodes, num_nodes))
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                adjacency_matrix[i, j] = 1 / (1 + abs(regional_scores[i, 0] - regional_scores[j, 0]))
    
    return adjacency_matrix

class PositionalEmbedding(nn.Module):
    def __init__(self, eeg_n_freqs, d_model, dropout=0.1, max_len=128):
        super().__init__()
        self.embed = nn.Linear(eeg_n_freqs, d_model)
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Ensure we are using the correct sequence length dimension of x
        x = self.embed(x)
        x = x + self.pe[:, :x.size(0), :]
        return self.dropout(x).squeeze(0)


class EEGScorer(nn.Module):
    def __init__(self):
        super().__init__()
        
        # This transformer encoder can encode important frequencies that relate to dementia by applying attention to the power
        # density over different frequencies and use this to differentiate dementia-affected regions.
        
        self.pos_embedding = PositionalEmbedding(eeg_n_freqs=1025, d_model=512)
        t_enc_layer = nn.TransformerEncoderLayer(d_model=512, nhead=4)
        self.psd_attn = nn.TransformerEncoder(t_enc_layer, num_layers=4)
        
        self.gin_scorer = GIN(
            in_channels=512,
            hidden_channels=256,
            out_channels=1,
            num_layers=2
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, eeg_graph, mmse):
        pos_embedded_psd = self.pos_embedding(eeg_graph.x)
        attended_psd_eeg = self.psd_attn(pos_embedded_psd)
        
        directed_edge_index = torch.combinations(torch.arange(19), r=2, with_replacement=False).T
        gin_psd_features_edge_index = torch.cat([directed_edge_index, directed_edge_index.flip(0)], dim=1)
        
        gin_scored = self.gin_scorer(attended_psd_eeg, gin_psd_features_edge_index)        
        mmse_conditioned_scores = torch.add(gin_scored, (1 - mmse))
        if torch.eq(mmse, 30):
            mmse_conditioned_scores = torch.zeros_like(mmse_conditioned_scores)
        
        gin_scored_norm = self.sigmoid(mmse_conditioned_scores).reshape(19, 1)
        return gin_scored_norm

class DementiaPredLossContext(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.gat = GAT(
            in_channels=1,
            hidden_channels=64,
            out_channels=128,
            num_layers=2
        )
        
        self.struct_latent_comp = GIN(
            in_channels=128,
            hidden_channels=256,
            out_channels=128,
            num_layers=2
        )
        
        self.mmse_proj = nn.Linear(1, 32)
        # TODO
        self.dem_class = nn.Linear(2432, 1)
        self.sigmoid = nn.Sigmoid()

        
    def forward(self, eeg_dem_scores):
        neural_scoring_coherence = torch.tensor(compute_neural_score_adj(eeg_dem_scores), dtype=torch.float)
        # print(neural_scoring_coherence.shape)
        edge_index_gat, edge_weight_gat = dense_to_sparse(neural_scoring_coherence)
        
        gat_features = self.gat(eeg_dem_scores, edge_index_gat, edge_weight_gat)
        graph_features = self.struct_latent_comp(gat_features, edge_index_gat)
        
        # print(gat_features.shape)
        # print(mmse_context.shape)
        
        flat_graph_features = graph_features.reshape(1, 19 * 128)
        
        dem_pred = self.dem_class(flat_graph_features)
        return self.sigmoid(dem_pred)
