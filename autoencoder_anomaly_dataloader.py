import mne
from torch.utils.data import Dataset, DataLoader
import torch
import os

from torch_geometric.utils import dense_to_sparse
from eeg_graph_construction import eeg_sim_matrix_calc

import matplotlib.pyplot as plt

ch_order = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']

# 0 = normal, 1 = dementia
def get_single_sample_eeg(file_path):
    raw = mne.io.read_raw_eeglab(file_path, preload=True)
    # raw.plot_psd_topomap()
    # print("EEG Channel Order:", raw.ch_names)
    eeg_signals, times = raw.get_data(return_times=True)
    eeg_signals = eeg_signals[:, :4096]
    # print(eeg_signals.shape)
    return eeg_signals

class EEGNormalDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        
        # Only focus on the 'normal' class
        self.classes = ['Normal']
        self.data = []

        for class_name in self.classes:
            class_folder = os.path.join(root_dir, class_name)
            for patient_folder in os.listdir(class_folder):
                patient_path = os.path.join(class_folder, patient_folder)
                eeg_folder_path = os.path.join(patient_path, 'eeg') 

                if os.path.isdir(eeg_folder_path):
                    for eeg_file in os.listdir(eeg_folder_path):
                        if eeg_file.endswith('.set'): 
                            eeg_file_path = os.path.join(eeg_folder_path, eeg_file)
                            self.data.append(eeg_file_path)  
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        eeg_file_path = self.data[idx]
        print(eeg_file_path)
        
        eeg_signal = torch.tensor(get_single_sample_eeg(eeg_file_path), dtype=torch.float)
        adj = torch.tensor(eeg_sim_matrix_calc(eeg_signal, sfreq=500), dtype=torch.float)
        eeg_index, eeg_attr = dense_to_sparse(adj)

        eeg_index = torch.tensor(eeg_index, dtype=torch.int64)
        eeg_attr = torch.tensor(eeg_attr, dtype=torch.float)

        return eeg_signal, eeg_index, eeg_attr

# get_single_sample_eeg('EEGNeurodegenerationDatasetClassed/Normal/sub-052/eeg/sub-052_task-eyesclosed_eeg.set')
root_dir = 'EEGNeurodegenerationDatasetClassed' 
batch_size = 1

dataset = EEGNormalDataset(root_dir=root_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# for node, idx, attr in dataloader:
#     print(node.shape)
#     print(idx.shape)
#     print(attr.shape)
