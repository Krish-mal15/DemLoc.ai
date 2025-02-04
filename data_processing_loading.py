import mne
from torch.utils.data import Dataset, DataLoader
import torch

import os
import pandas as pd
import numpy as np

from torch_geometric.utils import dense_to_sparse

from nilearn.connectome import ConnectivityMeasure
from mne.viz import plot_source_estimates

import networkx as nx
import matplotlib.pyplot as plt

participant_data = pd.read_csv('EEG-Dementia-Dataset/participants.tsv', sep='\t')
# print(participant_data)

ch_order = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']

# 0 = normal, 1 = dementia
def get_single_sample_eeg(file_path):
    raw = mne.io.read_raw_eeglab(file_path, preload=True)
    psd = raw.compute_psd().get_data()
    # plt.show()
    eeg_signals, times = raw.get_data(return_times=True)
    eeg_signals = eeg_signals[:, :4096]
    
    eeg_signals = torch.tensor(eeg_signals, dtype=torch.float).permute(1, 0).unsqueeze(0).numpy()
    
    # print(eeg_signals.shape)
    
    conn_measure = ConnectivityMeasure(kind='correlation')  
    conn_matrix = conn_measure.fit_transform(eeg_signals)
    # print(conn_matrix)
    
    conn_matrix = torch.tensor(conn_matrix, dtype=torch.float).squeeze(0)
        
    return psd, conn_matrix

# get_single_sample_eeg('EEGNeurodegenerationDatasetClassed/Dementia/sub-082/eeg/sub-082_task-eyesclosed_eeg.set')

class EEGDementiaDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        
        self.classes = ['Dementia', 'Normal']
        self.data = []

        for label, class_name in enumerate(self.classes):
            class_folder = os.path.join(root_dir, class_name)
            for patient_folder in os.listdir(class_folder):
                patient_path = os.path.join(class_folder, patient_folder)
                eeg_folder_path = os.path.join(patient_path, 'eeg') 
                
                if os.path.isdir(eeg_folder_path):
                    for eeg_file in os.listdir(eeg_folder_path):
                        if eeg_file.endswith('.set'):  # Check for .set files
                            eeg_file_str = eeg_file[:7]
                            # print(eeg_file_str)
                            
                            filtered_row = participant_data[participant_data['participant_id'] == eeg_file_str]
                            if not filtered_row.empty:
                                mmse_val = filtered_row['MMSE'].iloc[0]
                            
                                # print('')                         
                                # print(eeg_file)
                                # print(mmse_val)
                                # print('')
                                
                                norm_mmse = mmse_val / 30
                                eeg_file_path = os.path.join(eeg_folder_path, eeg_file)
                                # self.data.append((eeg_file_path, label))
                                self.data.append((eeg_file_path, norm_mmse, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # print(self.data)
        eeg_file_path, mmse, label = self.data[idx]
        # print(eeg_file_path)
        
        psd, eeg_conn = get_single_sample_eeg(eeg_file_path)
        psd = torch.tensor(psd, dtype=torch.float)
         
        eeg_index, eeg_attr = dense_to_sparse(eeg_conn)

        eeg_index = torch.tensor(eeg_index, dtype=torch.int64)
        eeg_attr = torch.tensor(eeg_attr, dtype=torch.float)

        return psd, eeg_index, eeg_attr, torch.tensor(mmse, dtype=torch.float), label

get_single_sample_eeg('EEGNeurodegenerationDatasetClassed/Normal/sub-052/eeg/sub-052_task-eyesclosed_eeg.set')
root_dir = 'EEGNeurodegenerationDatasetClassed' 
batch_size = 1

dataset = EEGDementiaDataset(root_dir=root_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# for node, idx, attr, mmse, label in dataloader:
#     print(node.shape)
#     print(idx.shape)
#     print(attr.shape)
#     print(mmse)
#     print(label)
#     print('')
