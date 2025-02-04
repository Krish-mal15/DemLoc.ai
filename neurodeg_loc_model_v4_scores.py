# Instead of analyzing gradients or attention, this will directly compute scores for each node which will be
# determine while backpropagating right and wrong dementia predictions.

from torch_geometric.nn import GIN, GATv2Conv
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data

import torch
import numpy as np 
import torch.nn as nn

from eeg_graph_construction import eeg_sim_matrix_calc
from data_processing_loading import get_single_sample_eeg

device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))


class DemLocalization(nn.Module):
    def __init__(self, num_eeg_channels, num_eeg_timesteps, latent_features):
        super().__init__()
        
        
        self.latent_features = latent_features
        self.num_eeg_channels = num_eeg_channels
        
      
        self.struct_featurizer = GIN(
            in_channels=num_eeg_timesteps,
            hidden_channels=512,
            out_channels=latent_features,
            num_layers=2
        )
        
        self.gin_scorer = GIN(
            in_channels=latent_features,
            hidden_channels=512,
            out_channels=1,
            num_layers=2
        )
                
        self.dementia_pred = nn.Linear(num_eeg_channels * latent_features, 1)
        
        self.sigmoid = nn.Sigmoid()  
        self.sigmoid_scores = nn.Sigmoid()  

        
    def forward(self, eeg_nodes, eeg_idx, eeg_attr):
        # Laent features should contain PLV (functional connectivity) information now
        neurodeg_feature_pred = self.struct_featurizer(x=eeg_nodes, edge_index=eeg_idx, edge_weight=eeg_attr)  # (19, latent_dim)
        
        # Dont need to provide edge weight because latent features should contain features about PLV
        region_scores = self.sigmoid_scores(self.gin_scorer(neurodeg_feature_pred, edge_index=eeg_idx))  # (19, 1)

        # print(neurodeg_feature_pred.shape)
        neurodeg_feature_pred = neurodeg_feature_pred.reshape(1, self.num_eeg_channels * self.latent_features)  # Same thing as squeezing, not sure though so this is safer
        dementia_pred = self.dementia_pred(neurodeg_feature_pred)
        
        dementia_pred = self.sigmoid(dementia_pred)  # Will change if using BCE with or without logits
        return dementia_pred, region_scores


model = DemLocalization(num_eeg_channels=19, num_eeg_timesteps=4096, latent_features=1024).to(device)

# # # --- Model Arbitrary Testing --- #
# test_eeg_signal = torch.tensor(get_single_sample_eeg('EEGNeurodegenerationDatasetClassed/Dementia/sub-003/eeg/sub-003_task-eyesclosed_eeg.set'), dtype=torch.float)
# test_adj = torch.tensor(eeg_sim_matrix_calc(test_eeg_signal, sfreq=500), dtype=torch.float)
# test_eeg_index, test_eeg_attr = dense_to_sparse(test_adj)

# test_eeg_index = torch.tensor(test_eeg_index, dtype=torch.int64)
# test_eeg_attr = torch.tensor(test_eeg_attr, dtype=torch.float)

# # print("Nodes: ", test_eeg_signal)
# # print("Index: ", test_eeg_index)

# test_eeg_graph = Data(x=test_eeg_signal, edge_index=test_eeg_index, edge_attr=test_eeg_attr)

# pred_dementia, scores = model(test_eeg_graph.x, test_eeg_graph.edge_index, test_eeg_graph.edge_attr)
# print(pred_dementia)
# print(scores)
