# Instead of analyzing gradients or attention, this will directly compute scores for each node which will be
# determine while backpropagating right and wrong dementia predictions.

from torch_geometric.nn import GIN, GATv2Conv
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer

import torch
import numpy as np 
import torch.nn as nn

from eeg_graph_construction import eeg_sim_matrix_calc
from data_processing_loading import get_single_sample_eeg

device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))


class DemLocalization(nn.Module):
    def __init__(self, num_eeg_channels, num_eeg_timesteps, latent_features, scoring_method):
        super().__init__()
        
        self.scoring_method = scoring_method
        
        self.latent_features = latent_features
        self.num_eeg_channels = num_eeg_channels
        
      
        self.struct_featurizer = GIN(
            in_channels=num_eeg_timesteps,
            hidden_channels=512,
            out_channels=latent_features,
            num_layers=2
        )
                
        self.gat_scorer1 = GATv2Conv(
            in_channels=latent_features,
            out_channels=latent_features,
            heads=4
        )
        self.gat_scorer2 = GATv2Conv(
            in_channels=latent_features * 4,
            out_channels=1,
            heads=1
        )
        self.regional_scores = nn.Linear(latent_features, 1)

        self.dementia_pred = nn.Linear(num_eeg_channels * latent_features, 1)
        
        self.sigmoid = nn.Sigmoid()  # Without Logits Loss
        self.relu = nn.ReLU()
        
    def forward(self, eeg_nodes, eeg_idx):
        neurodeg_feature_pred = self.struct_featurizer(eeg_nodes, eeg_idx) 
        
        if self.scoring_method == 'gat':
            print(neurodeg_feature_pred.shape)
            region_latent_features = self.gat_scorer1(neurodeg_feature_pred, eeg_idx)
            region_scores = self.gat_scorer2(region_latent_features, eeg_idx)
            # region_scores = self.sigmoid(region_scores)
        elif self.scoring_method == 'lin':
            region_scores = self.regional_scores(neurodeg_feature_pred)

        # print(neurodeg_feature_pred.shape)
        neurodeg_feature_pred = neurodeg_feature_pred.reshape(1, self.num_eeg_channels * self.latent_features)  # 4 because thats the number of attention heads
        dementia_pred = self.dementia_pred(neurodeg_feature_pred)
        
        dementia_pred = self.sigmoid(dementia_pred)  # Will change if using BCE with or without logits
        return dementia_pred, region_scores


# model = DemLocalization(num_eeg_channels=19, num_eeg_timesteps=4096, latent_features=128, scoring_method='gat').to(device)

# # # --- Model Arbitrary Testing --- #
# test_eeg_signal = torch.tensor(get_single_sample_eeg('EEGNeurodegenerationDatasetClassed/Dementia/sub-003/eeg/sub-003_task-eyesclosed_eeg.set'), dtype=torch.float)
# test_adj = torch.tensor(eeg_sim_matrix_calc(test_eeg_signal, sfreq=500), dtype=torch.float)
# test_eeg_index, test_eeg_attr = dense_to_sparse(test_adj)

# test_eeg_index = torch.tensor(test_eeg_index, dtype=torch.int64)
# test_eeg_attr = torch.tensor(test_eeg_attr, dtype=torch.float)

# # print("Nodes: ", test_eeg_signal)
# # print("Index: ", test_eeg_index)

# test_eeg_graph = Data(x=test_eeg_signal, edge_index=test_eeg_index, edge_attr=test_eeg_attr)

# pred_dementia, scores = model(test_eeg_graph.x, test_eeg_graph.edge_index)
# print(pred_dementia)
# print(scores)
