from torch_geometric.nn import GIN
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer

import torch
import numpy as np 
import torch.nn as nn

from eeg_graph_construction import eeg_sim_matrix_calc
from data_processing_loading import get_single_sample_eeg

device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

# Input nodes should still be (32, t), its just that t only contains power (PSD) from the alpha frequency band
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
        
        self.dementia_pred = nn.Linear(num_eeg_channels * latent_features, 1)
        self.sigmoid = nn.Sigmoid()  # Without Logits Loss
        
    def forward(self, eeg_nodes, eeg_idx):
        neurodeg_feature_pred = self.struct_featurizer(eeg_nodes, eeg_idx)  # struct_features are the new nodes

        # print(neurodeg_feature_pred.shape)
        neurodeg_feature_pred = neurodeg_feature_pred.reshape(1, self.num_eeg_channels * self.latent_features)  # 4 because thats the number of attention heads
        dementia_pred = self.dementia_pred(neurodeg_feature_pred)
        
        dementia_pred = self.sigmoid(dementia_pred)  # Will change if using BCE with or without logits
        return dementia_pred


# model = DemLocalization(num_eeg_channels=19, num_eeg_timesteps=1024, latent_features=128).to(device)
# explainer = Explainer(
#     model=model,
#     algorithm=GNNExplainer(epochs=200),
#     explanation_type='model',
#     node_mask_type='attributes',
#     edge_mask_type='object',
#     model_config=dict(
#         mode='binary_classification',
#         task_level='node',
#         return_type='raw',
#     ),
# )

# # --- Model Arbitrary Testing --- #
# test_eeg_signal = torch.tensor(get_single_sample_eeg('EEG-Dementia-Dataset/sub-066/eeg/sub-066_task-eyesclosed_eeg.set'), dtype=torch.float)
# test_adj = torch.tensor(eeg_sim_matrix_calc(test_eeg_signal, sfreq=500), dtype=torch.float)
# test_eeg_index, test_eeg_attr = dense_to_sparse(test_adj)

# test_eeg_index = torch.tensor(test_eeg_index, dtype=torch.int64)
# test_eeg_attr = torch.tensor(test_eeg_attr, dtype=torch.float)

# # print("Nodes: ", test_eeg_signal)
# # print("Index: ", test_eeg_index)

# test_eeg_graph = Data(x=test_eeg_signal, edge_index=test_eeg_index, edge_attr=test_eeg_attr)

# node_index = 0
# explanation = explainer(test_eeg_graph.x, test_eeg_graph.edge_index, index=node_index)
# print(f'Generated explanations in {explanation.available_explanations}')

# print(explanation.available_explanations)

# path = 'feature_importance.png'
# explanation.visualize_feature_importance(path, top_k=10)
# print(f"Feature importance plot has been saved to '{path}'")

# path = 'subgraph.pdf'
# explanation.visualize_graph(path)
# print(f"Subgraph visualization plot has been saved to '{path}'")

# model = DemLocalization(num_eeg_channels=19, num_eeg_timesteps=1024, latent_features=128)
# pred_dementia = model(test_eeg_graph)
# print(pred_dementia)
