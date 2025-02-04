import torch
from neurodeg_anomaly_ae import GraphAnomalyAE
from data_processing_loading import get_single_sample_eeg, ch_order
from torch_geometric.utils import dense_to_sparse
from eeg_graph_construction import eeg_sim_matrix_calc
from torch_geometric.data import Data

import numpy as np
import matplotlib.pyplot as plt
from mne.viz import plot_topomap
from mne.channels import make_standard_montage
from mne import create_info

from neurodeg_anomaly_ae_training import EEGSignalLoss

device = torch.device('cpu')

model = GraphAnomalyAE(device).to(device)
model.load_state_dict(torch.load("models_anomaly_graph_ae/dem_loc_model_epoch_12_anomaly_ae.pth", weights_only=True, map_location=device))

model.eval()

crit = EEGSignalLoss()

def get_eeg_graph(eeg_file_path):
    eeg_signal = torch.tensor(get_single_sample_eeg(eeg_file_path), dtype=torch.float)
    adj = torch.tensor(eeg_sim_matrix_calc(eeg_signal, sfreq=500), dtype=torch.float)
    eeg_index, eeg_attr = dense_to_sparse(adj)

    eeg_index = torch.tensor(eeg_index, dtype=torch.int64)
    eeg_attr = torch.tensor(eeg_attr, dtype=torch.float)
    
    eeg_graph = Data(x=eeg_signal, edge_index=eeg_index, eeg_attr=eeg_attr)

    return eeg_graph

def run_model(eeg_graph):
    with torch.no_grad():
        model_pred, latent_features = model(eeg_graph)
        
        # print(model_pred)
        
    return model_pred.squeeze(0)
    
eeg_graph = get_eeg_graph('EEGNeurodegenerationDatasetClassed/Dementia/sub-003/eeg/sub-003_task-eyesclosed_eeg.set')
model_pred = run_model(eeg_graph)

model_pred = model_pred.detach().numpy()
eeg_graph.x = eeg_graph.x.detach().numpy()

error = torch.tensor(np.abs(model_pred.mean(axis=1) - eeg_graph.x.mean(axis=1)), dtype=torch.float)
print("Error: ", error)

important_nodes = torch.argsort(error, descending=True)  # Rank nodes by importance

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
min_val = error.min()
max_val = error.max()

normalized = (error - min_val) / (max_val - min_val)
# print(normalized)

# Plotting the EEG channel importance topomap
plot_topomap(normalized, positions_2d, names=ch_order, 
             cmap='coolwarm', contours=0)
plt.title('EEG Channel Importance for Dementia Prediction')
plt.colorbar(label='Importance Score')
plt.show()

# error = crit(model_pred, eeg_graph.x)
# print(error)