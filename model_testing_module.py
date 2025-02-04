import torch
from neurodeg_context_scoring_v2 import EEGScorer
from data_processing_loading import get_single_sample_eeg, ch_order
from torch_geometric.utils import dense_to_sparse
from eeg_graph_construction import eeg_sim_matrix_calc
from torch_geometric.data import Data

import numpy as np
import matplotlib.pyplot as plt
from mne.viz import plot_topomap
from mne.channels import make_standard_montage
from mne import create_info

device = torch.device('cpu')

model = EEGScorer().to(device)
model.load_state_dict(torch.load("dem_loc_model_main_scorer_epoch_5_v2.pth", weights_only=True, map_location=device))

model.eval()

def get_eeg_graph(eeg_file_path):
    psd, corr = get_single_sample_eeg(eeg_file_path)
    eeg_index, eeg_attr = dense_to_sparse(corr)

    eeg_index = torch.tensor(eeg_index, dtype=torch.int64)
    eeg_attr = torch.tensor(eeg_attr, dtype=torch.float)
    
    # eeg_graph = Data(x=eeg_signal, edge_index=eeg_index, eeg_attr=eeg_attr)

    return torch.tensor(psd, dtype=torch.float), eeg_index, eeg_attr

def run_model(eeg_nodes, eeg_idx, eeg_attr, mmse):
    eeg_graph = Data(eeg_nodes, eeg_idx, eeg_attr)
    norm_mmse = mmse / 30
    norm_mmse_to_use = torch.tensor(norm_mmse, dtype=torch.float)
    with torch.no_grad():
        region_scores = model(eeg_graph, norm_mmse_to_use)
        
        # print(region_scores)
        
    return region_scores.squeeze(0)
    
eeg_nodes, eeg_idx, eeg_attr = get_eeg_graph('EEGNeurodegenerationDatasetClassed/Dementia/sub-066/eeg/sub-066_task-eyesclosed_eeg.set')
region_scores = run_model(eeg_nodes, eeg_idx, eeg_attr, 30)

print("Raw Scores: ", region_scores)
node_importance = region_scores.abs().sum(dim=1)  # Sum of absolute gradients for each node
print("Node importance: ", node_importance)
important_nodes = torch.argsort(node_importance, descending=True)  # Rank nodes by importance

# node_importance  = node_importance
# print("Model Prediction:", model_pred)
# print("Node Importance Scores:", node_importance)
print("Important Nodes (Ranked):", important_nodes)

important_ch = []
for node in important_nodes.numpy():
    important_ch.append(ch_order[node])
    
print("Important Channels: ", important_ch)

montage = make_standard_montage('standard_1020')

# Create an MNE Info object for channels
info = create_info(ch_order, sfreq=500, ch_types="eeg")
info.set_montage(montage)

# Get 2D positions for plotting
positions_2d = np.array([montage.get_positions()['ch_pos'][ch][:2] for ch in ch_order])

# Normalize node importance
min_val = node_importance.min()
max_val = node_importance.max()

normalized = (node_importance - min_val) / (max_val - min_val)
# print(normalized)

# Plotting the EEG channel importance topomap
plot_topomap(normalized, positions_2d, names=ch_order, 
             cmap='coolwarm', contours=0)
plt.title('EEG Channel Importance for Dementia Prediction')
plt.colorbar(label='Importance Score')
plt.show()
