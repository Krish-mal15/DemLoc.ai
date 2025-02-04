import shap
import torch
from neurodeg_loc_model_v2 import DemLocalization
from data_processing_loading import get_single_sample_eeg, ch_order
from torch_geometric.utils import dense_to_sparse
from eeg_graph_construction import eeg_sim_matrix_calc
from torch_geometric.data import Data

device = torch.device('cpu')

# Load the model
model = DemLocalization(num_eeg_channels=19, num_eeg_timesteps=4096, latent_features=128).to(device)
model.load_state_dict(torch.load("models/dem_loc_model_epoch_50.pth", weights_only=True, map_location=device))
model.eval()

# Function to construct the EEG graph
def get_eeg_graph(eeg_file_path):
    eeg_signal = torch.tensor(get_single_sample_eeg(eeg_file_path), dtype=torch.float)
    adj = torch.tensor(eeg_sim_matrix_calc(eeg_signal, sfreq=500), dtype=torch.float)
    eeg_index, _ = dense_to_sparse(adj)
    return eeg_signal, eeg_index

# Prepare the input data
eeg_nodes, eeg_idx = get_eeg_graph('EEG-Dementia-Dataset/sub-017/eeg/sub-017_task-eyesclosed_eeg.set')
data = Data(x=eeg_nodes, edge_index=eeg_idx)

node_idx = 0  # Example node index

# Define a model wrapper
def model_node_wrapper(node_features):
    # Convert input to PyTorch tensor with batch dimension
    node_features = torch.tensor(node_features, dtype=torch.float).unsqueeze(0)  # Add batch dim
    with torch.no_grad():
        output = model(node_features.to(device), data.edge_index.to(device))
    return output[0].detach().numpy()  # Return the prediction for the node

# Use the features of the selected node
node_features = data.x[node_idx].unsqueeze(0).numpy()

# Create the SHAP explainer
explainer = shap.Explainer(model_node_wrapper, node_features)

# Compute SHAP values for the selected node
shap_values = explainer(node_features)

# Visualize the results
shap.plots.waterfall(shap_values[0])