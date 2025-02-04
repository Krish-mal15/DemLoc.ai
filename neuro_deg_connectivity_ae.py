import torch.nn as nn
import torch
import numpy as np

import networkx as nx
import matplotlib.pyplot as plt

from torch_geometric.nn import GATv2Conv, GIN
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data

from eeg_graph_construction import eeg_sim_matrix_calc
from data_processing_loading import get_single_sample_eeg
device = torch.device('cpu')

# Input should just be a tensor (float) of the MMSE value between 0-1, mapped from the 0-30 range using minmax normalization
# This class will just project the tensor into a latent space to condition the graph latent space
        

# Latent dimension of the autoencoder should be larger since its raw features and not an expressive distribution.
class GraphConnectionEncoder(nn.Module):
    def __init__(self, n_eeg_timesteps, latent_dim, mmse_condition):
        super().__init__()
        
        self.mmse_condition = mmse_condition
        
        self.gat_encoder = GATv2Conv(
            in_channels=n_eeg_timesteps,
            out_channels=latent_dim,
        )
        
        self.gat_encoder_2 = GATv2Conv(
            in_channels=latent_dim,
            out_channels=latent_dim
        )
        
        self.mmse_proj = nn.Linear(1, latent_dim)
         
    def forward(self, eeg_graph, mmse_score):
        attn_features1, attn = self.gat_encoder(eeg_graph.x, eeg_graph.edge_index, return_attention_weights=True)
        idx, attn_weights = attn
        attn_weights = torch.tensor(attn_weights, dtype=torch.float).squeeze(1)
        print("ATTN: ", attn_weights.shape)
        # The output of attention features 1 wil include attention weights from the power spectral density of individual brain regions
        graph_features, attn = self.gat_encoder_2(attn_features1, eeg_graph.edge_index, attn_weights)
        
        # print(graph_features)
        
        if self.mmse_condition:
            mmse_embedded = self.mmse_proj(mmse_score)
            # print(graph_features)
            # print(mmse_embedded)
            # print(mmse_embedded.shape)
            # print(graph_features.shape)
            graph_features += mmse_embedded
            # print(graph_features)
        
        return graph_features, attn_weights # (19, latent_dim)
    
# For theoretically better propagation, the below model will classify a predicted connectivity matrix as dementia or not
# use graph isomorphism and then this can condition the connectivity predictions to be based on the dementia class or not.
class ConnectivityClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.gin = GIN(
            in_channels=1025,
            hidden_channels=256,
            out_channels=512,
            num_layers=2
        )
        
        self.predictor = nn.Linear(512 * 19, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, eeg_graph, pred_connectivity):
        gin_encoded = self.gin(eeg_graph.x, eeg_graph.edge_index, pred_connectivity)
        pred_dem = self.sigmoid(self.predictor(gin_encoded.reshape(19*512)))
        
        return pred_dem
    
class GraphConnectivityDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = GraphConnectionEncoder(n_eeg_timesteps=1025, latent_dim=512, mmse_condition=True)
        self.classifier_conditioning = ConnectivityClassifier()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, eeg_graph, mmse):
        latent_graph_features, attn_weights = self.encoder(eeg_graph, mmse)
        decoded_out = torch.matmul(latent_graph_features, latent_graph_features.T)
        compressed_out = self.sigmoid(decoded_out)
        pred_dem = self.classifier_conditioning(eeg_graph, compressed_out)
        print(pred_dem)
        
        # avg_attn = torch.mean(attn_weights, dim=1)
    
        return compressed_out, attn_weights


model = GraphConnectivityDecoder()

# # # --- Model Arbitrary Testing --- #
test_psd, test_eeg_signal = get_single_sample_eeg('EEGNeurodegenerationDatasetClassed/Dementia/sub-003/eeg/sub-003_task-eyesclosed_eeg.set')
test_psd = torch.tensor(test_psd, dtype=torch.float)
test_eeg_signal = torch.tensor(test_eeg_signal, dtype=torch.float)

test_adj = torch.tensor(eeg_sim_matrix_calc(test_eeg_signal, sfreq=500), dtype=torch.float)
test_eeg_index, test_eeg_attr = dense_to_sparse(test_adj)

test_eeg_index = torch.tensor(test_eeg_index, dtype=torch.int64)
test_eeg_attr = torch.tensor(test_eeg_attr, dtype=torch.float)

test_eeg_graph = Data(x=test_psd, edge_index=test_eeg_index, edge_attr=test_eeg_attr)
# print("EEG Graph: ", test_eeg_graph)

test_mmse = torch.tensor(np.random.rand(1, 1), dtype=torch.float)
# print(test_mmse)

decoded, attn = model(test_eeg_graph, test_mmse)
decoded_mtx = decoded.reshape(19, 19).detach().numpy()

test_true_mtx = test_eeg_attr.reshape(19, 19).detach().numpy()

plt.figure(figsize=(8, 8))
plt.imshow(test_true_mtx, cmap='Blues', interpolation='none')
plt.colorbar(label='Connection Strength')
plt.title('19x19 Adjacency Matrix')
plt.xlabel('Nodes')
plt.ylabel('Nodes')
plt.xticks(range(19))
plt.yticks(range(19))
# plt.show()
print(decoded)
print(attn.shape)
        
