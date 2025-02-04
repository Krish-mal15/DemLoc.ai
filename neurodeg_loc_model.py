from torch_geometric.nn import GIN, GATv2Conv
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data

import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F

from data_processing_loading import get_single_sample_eeg
from eeg_graph_construction import eeg_sim_matrix_calc


class NeuroDegGraphLocalization(nn.Module):
    def __init__(self, num_eeg_channels, num_eeg_timesteps, latent_features):
        super().__init__()
        
        self.latent_features = latent_features
        self.num_eeg_channels = num_eeg_channels
        
        # Add batch norm after GIN to normalize features
        self.struct_featurizer = GIN(
            in_channels=num_eeg_timesteps,
            hidden_channels=512,
            out_channels=latent_features,
            num_layers=2
        )
        self.batch_norm = nn.BatchNorm1d(latent_features)
        
        # Modified GAT layer with proper initialization
        self.attn_featurizer = GATv2Conv(
            in_channels=latent_features,
            out_channels=latent_features // 4,  # Reduce output channels since we have 4 heads
            heads=4,
            add_self_loops=False,
            dropout=0.1  # Add dropout for regularization
        )
        
        # Add layer norm before final prediction
        self.layer_norm = nn.LayerNorm(num_eeg_channels * latent_features)
        self.dementia_pred = nn.Linear(num_eeg_channels * latent_features, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, eeg_graph):
        eeg_nodes, eeg_idx, eeg_attr = eeg_graph.x, eeg_graph.edge_index, eeg_graph.edge_attr
        
        # Get structural features
        struct_features = self.struct_featurizer(eeg_nodes, eeg_idx, eeg_attr)
        struct_features = self.batch_norm(struct_features)
        
        # Apply attention mechanism
        neurodeg_feature_pred, node_wise_attention = self.attn_featurizer(
            struct_features, 
            eeg_idx,
            return_attention_weights=True
        )
        
        _, edge_gen_weights = node_wise_attention
        # Keep multi-head attention weights separate initially
        edge_gen_weights = edge_gen_weights.view(4, -1)  # [num_heads, num_edges]
        
        # Calculate node attention for each head separately
        node_attention = torch.zeros(4, self.num_eeg_channels).to(edge_gen_weights.device)
        for head in range(4):
            head_weights = edge_gen_weights[head]
            for i, edge in enumerate(eeg_idx.T):
                src, dest = edge
                node_attention[head, src] += head_weights[i]
                node_attention[head, dest] += head_weights[i]
        
        # Normalize node attention per head
        node_degree = torch.bincount(eeg_idx.flatten(), minlength=self.num_eeg_channels)
        node_attention = node_attention / (node_degree + 1e-9).unsqueeze(0)
        
        # Apply softmax to ensure weights sum to 1 for each head
        node_attention = F.softmax(node_attention, dim=1)
        
        # Reshape features and apply attention
        neurodeg_feature_pred = neurodeg_feature_pred.view(self.num_eeg_channels, -1)
        
        # Apply layer norm before final prediction
        neurodeg_feature_pred = self.layer_norm(neurodeg_feature_pred.flatten().unsqueeze(0))
        dementia_pred = self.dementia_pred(neurodeg_feature_pred)
        dementia_pred = self.sigmoid(dementia_pred)
        
        # Return mean attention across heads for analysis
        return dementia_pred, node_attention.mean(dim=0), edge_gen_weights.mean(dim=0)


# --- Model Arbitrary Testing --- #
# test_eeg_signal = torch.tensor(get_single_sample_eeg('EEG-Dementia-Dataset/sub-066/eeg/sub-066_task-eyesclosed_eeg.set'), dtype=torch.float)
# test_adj = torch.tensor(eeg_sim_matrix_calc(test_eeg_signal, sfreq=500), dtype=torch.float)
# test_eeg_index, test_eeg_attr = dense_to_sparse(test_adj)

# test_eeg_index = torch.tensor(test_eeg_index, dtype=torch.int64)
# test_eeg_attr = torch.tensor(test_eeg_attr, dtype=torch.float)

# # print("Nodes: ", test_eeg_signal)
# # print("Index: ", test_eeg_index)
# # print("Attr: ", test_eeg_attr)

# test_eeg_graph = Data(x=test_eeg_signal, edge_index=test_eeg_index, edge_attr=test_eeg_attr)
# model = NeuroDegGraphLocalization(num_eeg_channels=19, num_eeg_timesteps=1024, latent_features=128)

# pred_dementia, attn_weights, connection_weights = model(test_eeg_graph)

# print(pred_dementia)
# print(attn_weights)
